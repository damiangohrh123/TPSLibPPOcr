"""
Tesseract OCR Benchmark
========================
Feeds the full image to Tesseract and reports timing, CPU, memory, and
recognised text. Uses TSV output to extract per-word confidence scores
and bounding box coordinates.

Preprocessing: Otsu binarization (same as ppocr_benchmark.py).

Usage:
    python3 tesseract_benchmark.py --image_path test_images/full_test.jpeg
    python3 tesseract_benchmark.py --image_path test_images/full_test.jpeg --cycles 3
"""

import os
import sys
import time
import argparse
import subprocess
import tempfile
import cv2
import numpy as np


# ── env ───────────────────────────────────────────────────────────────────────

_ENV = os.environ.copy()
_ENV['LD_LIBRARY_PATH'] = '/usr/lib/tps'
_ENV['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata'


# ── /proc helpers (mirrors ppocr_benchmark.py) ───────────────────────────────

def get_rss_mb():
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        return 0.0
    return 0.0


def read_cpu_times():
    try:
        with open('/proc/self/stat', 'r') as f:
            fields = f.read().split()
        # fields[13]/[14] = utime/stime (this process)
        # fields[15]/[16] = cutime/cstime (waited-for children, i.e. tesseract subprocess)
        return int(fields[13]) + int(fields[15]), int(fields[14]) + int(fields[16])
    except Exception:
        return 0, 0


def read_system_cpu_total():
    try:
        with open('/proc/stat', 'r') as f:
            vals = [int(x) for x in f.readline().split()[1:]]
        return sum(vals)
    except Exception:
        return 0


def get_tesseract_version():
    try:
        r = subprocess.run(['tesseract', '--version'],
                           capture_output=True, text=True, env=_ENV)
        return r.stdout.splitlines()[0].strip()
    except Exception:
        return 'unknown'


# ── Tesseract TSV parser ──────────────────────────────────────────────────────

def run_tesseract(img_path):
    """
    Run Tesseract on a pre-written image file with TSV output.
    Returns list of (text, score, tl_x, tl_y) per word, filtered by conf >= 0.
    """
    result = subprocess.run(
        ['tesseract', img_path, 'stdout', '--dpi', '150', 'tsv'],
        capture_output=True, text=True, env=_ENV, timeout=30
    )
    if result.returncode != 0:
        return []

    words = []
    for line in result.stdout.splitlines()[1:]:   # skip header
        parts = line.split('\t')
        if len(parts) < 12:
            continue
        conf = int(parts[10])
        text = parts[11].strip()
        if conf < 0 or not text:
            continue
        left = int(parts[6])
        top  = int(parts[7])
        score = conf / 100.0
        words.append((text, score, left, top))
    return words


# ── benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(args):
    if not os.path.exists(args.image_path):
        print('ERROR: image not found: {0}'.format(args.image_path))
        sys.exit(1)

    img = cv2.imread(args.image_path)
    if img is None:
        print('ERROR: cannot read image: {0}'.format(args.image_path))
        sys.exit(1)

    print('\n  Tesseract Benchmark  ({0} cycles)'.format(args.cycles))
    print('  Image      : {0}'.format(args.image_path))
    print('  Engine     : {0}'.format(get_tesseract_version()))
    print('  Input scale: {0}x'.format(args.input_scale))
    print('  Drop score : {0}'.format(args.drop_score))
    print('--')

    # Binarize + optional upscale — same preprocessing as ppocr_benchmark.py
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if args.input_scale != 1.0:
        h, w = bw.shape[:2]
        bw = cv2.resize(bw,
                        (int(w * args.input_scale), int(h * args.input_scale)),
                        interpolation=cv2.INTER_LANCZOS4)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, bw)

    times_ms = []
    cpu_pcts = []
    mem_mbs  = []
    last_words = []

    print('Running {0} cycles...'.format(args.cycles))

    try:
        for cycle_idx in range(args.cycles):
            mem_before = get_rss_mb()
            u0, s0 = read_cpu_times()
            sys0 = read_system_cpu_total()
            wall_start = time.time()

            words = run_tesseract(tmp_path)

            wall_end = time.time()
            mem_after = get_rss_mb()
            u1, s1 = read_cpu_times()
            sys1 = read_system_cpu_total()

            elapsed_ms = (wall_end - wall_start) * 1000.0
            proc_delta = (u1 - u0) + (s1 - s0)
            sys_delta  = sys1 - sys0
            cpu_pct    = (proc_delta / sys_delta * 100.0 * os.cpu_count()) if sys_delta > 0 else 0.0

            times_ms.append(elapsed_ms)
            cpu_pcts.append(cpu_pct)
            mem_mbs.append(max(mem_before, mem_after))

            if cycle_idx == args.cycles - 1:
                last_words = words
    finally:
        os.unlink(tmp_path)

    t_arr = np.array(times_ms)
    c_arr = np.array(cpu_pcts)
    m_arr = np.array(mem_mbs)

    print('\n  RESULTS')
    print('--')
    print('  Avg time   : {0:.1f} ms  (std {1:.1f})'.format(t_arr.mean(), t_arr.std()))
    print('  Avg CPU    : {0:.1f}%'.format(c_arr.mean()))
    print('  Avg memory : {0:.1f} MB'.format(m_arr.mean()))
    print('  Recognized text:')
    if last_words:
        for text, score, tl_x, tl_y in last_words:
            if score < args.drop_score:
                continue
            print('    "{0}"  [{1:.2f}]  tl: ({2}, {3})'.format(
                text, score, tl_x, tl_y))
    else:
        print('    (none)')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Tesseract Benchmark')
    p.add_argument('--image_path',  type=str,   required=True)
    p.add_argument('--cycles',      type=int,   default=1)
    p.add_argument('--input_scale', type=float, default=2.0)
    p.add_argument('--drop_score',  type=float, default=0.5)
    run_benchmark(p.parse_args())
