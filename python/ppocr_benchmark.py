"""
PPOCR Benchmark — runs det + rec on an image N times and reports timing, CPU,
memory, and recognised text. Saves an annotated image to --out_dir.
Models are loaded once before the loop; timings reflect pure inference cost.

Usage:
    PPOCR_CHAR_DICT=../model/ppocr_keys_v6.txt \
    python3 ppocr_benchmark.py --image_path test_images/full_test.jpeg

Key env vars: PPOCR_CHAR_DICT, PPOCR_DET_THRESH, PPOCR_BOX_THRESH,
              PPOCR_UNCLIP_RATIO, PPOCR_DROP_SCORE, PPOCR_MIN_HEIGHT,
              PPOCR_MIN_WIDTH, PPOCR_BINARIZE
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import ppocr_det as predict_det
import ppocr_rec as predict_rec
from ppocr_system import (
    _run_det_tiled, _nms_boxes, _box_size_ok,
    get_rotate_crop_image, sorted_boxes, preprocess_image,
)


# ── /proc helpers (no psutil needed) ─────────────────────────────────────────

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
        return int(fields[13]), int(fields[14])
    except Exception:
        return 0, 0


def read_system_cpu_total():
    try:
        with open('/proc/stat', 'r') as f:
            vals = [int(x) for x in f.readline().split()[1:]]
        return sum(vals)
    except Exception:
        return 0


# ── benchmark ─────────────────────────────────────────────────────────────────

def save_crops(crops, rec_results, crops_dir):
    """Save each rec input crop as crops/<idx>_<text>_<score>.png."""
    os.makedirs(crops_dir, exist_ok=True)
    for i, (crop, res) in enumerate(zip(crops, rec_results)):
        text, score = res[0]
        # Sanitise text for use in filename
        safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in text)
        fname = '{0:02d}_{1}_{2:.2f}.png'.format(i, safe, score)
        cv2.imwrite(os.path.join(crops_dir, fname), crop)
    print('  Crops saved : {0}/  ({1} files)'.format(crops_dir, len(crops)))


def run_once(img_orig, detector, recogniser, input_scale, drop_score, cycles,
             save_crops_dir=None):

    # Compute constants once before the timing loop.
    _min_h   = float(os.environ.get('PPOCR_MIN_HEIGHT', '10')) * input_scale
    _min_w   = float(os.environ.get('PPOCR_MIN_WIDTH',  '8'))  * input_scale
    binarize = os.environ.get('PPOCR_BINARIZE', '1') != '0'

    times_ms = []
    cpu_pcts = []
    mem_mbs  = []
    last_res = None
    last_crops = []
    last_det_ms = last_rec_ms = last_n_crops = 0

    for cycle_idx in range(cycles):
        mem_before = get_rss_mb()
        u0, s0 = read_cpu_times()
        sys0 = read_system_cpu_total()
        wall_start = time.time()

        img_det = preprocess_image(img_orig, input_scale, binarize)

        _t_det_start = time.time()
        dt_boxes = _run_det_tiled(img_det, detector)
        _det_ms = (time.time() - _t_det_start) * 1000.0
        dt_boxes = sorted_boxes(dt_boxes) if dt_boxes is not None and len(dt_boxes) else []

        dt_boxes = [b for b in dt_boxes if _box_size_ok(b, _min_h, _min_w)]
        dt_boxes = _nms_boxes(dt_boxes)

        crops = [get_rotate_crop_image(img_det, b) for b in dt_boxes]
        _t_rec_start = time.time()
        rec_results = recogniser.run(crops) if crops else []
        _rec_ms = (time.time() - _t_rec_start) * 1000.0

        wall_end = time.time()
        # Keep crops from last cycle for optional saving (outside timed section)
        if cycle_idx == cycles - 1:
            last_crops = crops
            last_det_ms = _det_ms
            last_rec_ms = _rec_ms
            last_n_crops = len(crops)
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

        if input_scale != 1.0:
            dt_boxes = [b / input_scale for b in dt_boxes]
        last_res = list(zip(dt_boxes, rec_results))

    if save_crops_dir and last_crops and last_res:
        save_crops(last_crops, [r for _, r in last_res], save_crops_dir)

    t_arr = np.array(times_ms)
    c_arr = np.array(cpu_pcts)
    m_arr = np.array(mem_mbs)
    return {
        'last_res':    last_res,
        'avg_ms':      t_arr.mean(),
        'std_ms':      t_arr.std(),
        'avg_cpu':     c_arr.mean(),
        'avg_mem':     m_arr.mean(),
        'last_det_ms': last_det_ms,
        'last_rec_ms': last_rec_ms,
        'last_n_crops': last_n_crops,
    }


def print_and_save(label, stats, drop_score, out_dir, filename, img_orig):
    print('\n  {0}'.format(label))
    print('--')
    print('  Avg time   : {0:.1f} ms  (std {1:.1f})'.format(stats['avg_ms'], stats['std_ms']))
    other_ms = stats['avg_ms'] - stats['last_det_ms'] - stats['last_rec_ms']  # approx
    print('    Det      : {0:.1f} ms'.format(stats['last_det_ms']))
    print('    Rec      : {0:.1f} ms  ({1} crops)'.format(stats['last_rec_ms'], stats['last_n_crops']))
    print('    Other    : {0:.1f} ms  (pre/postprocess, NMS, crop extraction)'.format(other_ms))
    print('  Avg CPU    : {0:.1f}%'.format(stats['avg_cpu']))
    print('  Avg memory : {0:.1f} MB'.format(stats['avg_mem']))
    print('  Recognized text:')
    last_res = stats['last_res']
    if last_res:
        for box, res in last_res:
            text, score = res[0]
            if score < drop_score:
                continue
            center = box.mean(axis=0).astype(int)
            tl     = box[0].astype(int)
            print('    "{0}"  [{1:.2f}]  tl: ({2}, {3})  c: ({4}, {5})'.format(
                text, score, tl[0], tl[1], center[0], center[1]))
    else:
        print('    (none)')

    if last_res:
        os.makedirs(out_dir, exist_ok=True)
        annotated = img_orig.copy()
        for box, res in last_res:
            text, score = res[0]
            pts = np.array(box, dtype=np.int32)
            kept = score >= drop_score
            color = (0, 220, 0) if kept else (0, 0, 210)  # green=kept, red=dropped
            cv2.polylines(annotated, [pts], True, color, 2)
            ann_label = '{0}:{1:.2f}'.format(text[:10], score) if kept else '{0:.2f}'.format(score)
            cv2.putText(annotated, ann_label, tuple(pts[0].tolist()),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        ann_path = os.path.join(out_dir, filename)
        cv2.imwrite(ann_path, annotated)
        print('  Annotated  : {0}'.format(ann_path))


def run_benchmark(args):
    os.environ['PPOCR_DROP_SCORE'] = str(args.drop_score)

    model_args = argparse.Namespace(
        det_model_path=args.det_model,
        rec_model_path=args.rec_model,
        target=args.target,
        device_id=None,
    )

    img_name = os.path.splitext(os.path.basename(args.image_path))[0]
    rec_name = os.path.splitext(os.path.basename(args.rec_model))[0]

    print('\n  PPOCR Benchmark  ({0} cycles)'.format(args.cycles))
    print('  Image      : {0}'.format(args.image_path))
    print('  Det model  : {0}'.format(args.det_model))
    print('  Rec model  : {0}'.format(args.rec_model))
    print('  Drop score : {0}'.format(args.drop_score))
    print('  Input scale: {0}x'.format(args.input_scale))
    print('--')

    if not os.path.exists(args.image_path):
        print('ERROR: image not found: {0}'.format(args.image_path))
        sys.exit(1)

    img = cv2.imread(args.image_path)
    if img is None:
        print('ERROR: cannot read image: {0}'.format(args.image_path))
        sys.exit(1)

    print('Loading models (not included in timing)...')
    detector   = predict_det.TextDetector(model_args)
    recogniser = predict_rec.TextRecognizer(model_args)
    print('Models loaded.')

    crops_dir = args.crops_dir if args.save_crops else None

    print('\nRunning {0} cycles...'.format(args.cycles))
    stats = run_once(img, detector, recogniser, args.input_scale,
                     args.drop_score, args.cycles,
                     save_crops_dir=crops_dir)
    print_and_save('RESULTS  ({0}x)'.format(args.input_scale),
                   stats, args.drop_score, args.out_dir,
                   '{0}_{1}_{2}x.png'.format(img_name, rec_name, args.input_scale), img)

    # ── release ───────────────────────────────────────────────────────────────
    try:
        detector.model.release()
    except Exception:
        pass
    try:
        recogniser.model.release()
    except Exception:
        pass


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='PPOCR Benchmark')
    p.add_argument('--image_path',  type=str,   required=True)
    p.add_argument('--det_model',   type=str,   default='../model/PP-OCRv6_tiny_det_rk3588.rknn')
    p.add_argument('--rec_model',   type=str,   default='../model/PP-OCRv6_tiny_rec_rk3588.rknn')
    p.add_argument('--target',      type=str,   default='rk3588')
    p.add_argument('--cycles',      type=int,   default=1)
    p.add_argument('--input_scale', type=float, default=2.0)  # 2x = best det coverage
    p.add_argument('--drop_score',  type=float, default=0.4)
    p.add_argument('--out_dir',     type=str,   default='annotated')
    p.add_argument('--save_crops',   action='store_true',
                   help='Save each rec input crop to --crops_dir')
    p.add_argument('--crops_dir',    type=str,   default='crops')
    run_benchmark(p.parse_args())
