#ifndef TPS_PPOCR_H
#define TPS_PPOCR_H

// TPSLibPPOcr: PP-OCR text reading on the RK3588 NPU, as a flat C interface (see README.md, "Production Use").
// Frames are raw BGR888 rows. A handle may be shared by threads; its calls run one at a time.
// Each result belongs to the caller until tps_ppocr_free. No function throws or exits; failures return a TPS_PPOCR_ERR_* code.
// The loading program must link libpthread, as app and appui do: on glibc before 2.34, C++ threads fail if it arrives only with this library.

#define TPS_PPOCR_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

#define TPS_PPOCR_ABI_VERSION 1

#define TPS_PPOCR_OK 0
#define TPS_PPOCR_ERR_INVALID_ARG -1  // a null pointer, a bad frame size or a region outside the frame
#define TPS_PPOCR_ERR_LOAD_FAILED -2  // a model or the character list could not be loaded
#define TPS_PPOCR_ERR_READ_FAILED -3  // the pipeline failed while reading

typedef struct tps_ppocr tps_ppocr;  // opaque: the loaded models

// One piece of text found in the frame.
typedef struct tps_ppocr_piece {
    const char* text;  // UTF-8
    float score;       // 0 to 1
    float box[8];      // corners x0, y0 ... x3, y3 in frame pixels: top-left, top-right, bottom-right, bottom-left
} tps_ppocr_piece;

// What one read found.
typedef struct tps_ppocr_result {
    const char* text;               // UTF-8; pieces on one line joined by spaces, lines by "\n", top to bottom
    int confidence;                 // mean piece score, 0 to 100; 0 when nothing was read
    int count;                      // number of pieces
    const tps_ppocr_piece* pieces;  // in reading order
} tps_ppocr_result;

// Returns TPS_PPOCR_ABI_VERSION, so a loader can check the library before calling anything else.
TPS_PPOCR_API int tps_ppocr_abi_version(void);

// Loads PP-OCRv6_tiny_det_rk3588.rknn, PP-OCRv6_tiny_rec_rk3588.rknn and ppocr_keys_v6.txt from model_dir.
TPS_PPOCR_API int tps_ppocr_create(const char* model_dir, tps_ppocr** out_handle);

// Reads a whole frame. stride is the bytes per row, at least width * 3.
TPS_PPOCR_API int tps_ppocr_read_frame(tps_ppocr* handle, const unsigned char* bgr, int width, int height, int stride,
    tps_ppocr_result** out_result);

// Reads one region drawn on a frame, as a Screen Capture region: only text centred inside the region is kept.
TPS_PPOCR_API int tps_ppocr_read_region(tps_ppocr* handle, const unsigned char* bgr, int width, int height, int stride,
    int x, int y, int region_width, int region_height, tps_ppocr_result** out_result);

// Frees a result from tps_ppocr_read_frame or tps_ppocr_read_region; NULL is ignored.
TPS_PPOCR_API void tps_ppocr_free(tps_ppocr_result* result);

// Releases the models; NULL is ignored. No call on the handle may be running.
TPS_PPOCR_API void tps_ppocr_destroy(tps_ppocr* handle);

// Describes the calling thread's last failed call.
TPS_PPOCR_API const char* tps_ppocr_last_error(void);

#ifdef __cplusplus
}
#endif

#endif  // TPS_PPOCR_H
