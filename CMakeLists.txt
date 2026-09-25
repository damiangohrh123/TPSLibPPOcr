cmake_minimum_required(VERSION 3.16)
project(tpslibppocr CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPENCV_INCLUDE_DIR "" CACHE PATH "Directory containing opencv2/*.hpp")
set(OPENCV_LIB_DIR     "" CACHE PATH "Directory containing libopencv_core.so etc.")
set(CLIPPER_INCLUDE_DIR "" CACHE PATH "Directory containing polyclipping/clipper.hpp")
set(CLIPPER_LIB_DIR     "" CACHE PATH "Directory containing libpolyclipping.so")
set(ZLIB_LIB_DIR        "" CACHE PATH "Directory containing libz (needed by opencv_core's persistence/FileStorage code even without imgcodecs)")

set(RKNN_SDK_LIB_DIR    "" CACHE PATH "Directory containing librknnrt.so (board SDK only)")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OPENCV_INCLUDE_DIR}
    ${CLIPPER_INCLUDE_DIR}
)
link_directories(${OPENCV_LIB_DIR} ${CLIPPER_LIB_DIR} ${ZLIB_LIB_DIR})

# ocr_core: the OCR logic (detector, recognizer, and
# shared helpers) as a static library, not a runnable program.
add_library(ocr_core STATIC
    ocr/rknn_executor.cpp
    ocr/preprocess.cpp
    ocr/ppocr_det.cpp
    ocr/ppocr_rec.cpp
    ocr/ppocr_system.cpp
)

target_include_directories(ocr_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# imgcodecs is intentionally not linked: frames arrive as already-decoded
# cv::Mat from an earlier pipeline stage, so dropping it avoids needing
# libjpeg/libpng for the board build. zlib is still required regardless --
# opencv_core's FileStorage code calls gz* internally.
#
# --start-group around core+imgproc: imgproc calls core-internal helpers our
# own code never references, which a single left-to-right pass would leave
# unresolved. -ldl must follow the group: opencv_core's plugin loader is the
# only caller of dlopen/dlsym, so the pending need only exists after core
# has been scanned.
target_link_libraries(ocr_core PUBLIC
    -Wl,--start-group opencv_core opencv_imgproc -Wl,--end-group
    dl polyclipping z pthread
)

if(RKNN_SDK_LIB_DIR)
    target_link_directories(ocr_core PUBLIC ${RKNN_SDK_LIB_DIR})
    target_link_libraries(ocr_core PUBLIC rknnrt)
endif()

# benchmark: one raw .bgr888 image in, OCR text and timing out. Built once a
# real librknnrt.so is available (RKNN_SDK_LIB_DIR set).
# librknnrt.so dynamically depends on the system's own libstdc++.so.6 at runtime
# (independent of ours being statically linked in), so benchmark passes
# -Wl,--allow-shlib-undefined: that lets it resolve at load time on the board
# instead of demanding a matching .so be present at link time here.
option(BUILD_TOOLS "Build the benchmark tool" OFF)
if(BUILD_TOOLS)
    if(NOT RKNN_SDK_LIB_DIR)
        message(FATAL_ERROR "BUILD_TOOLS requires RKNN_SDK_LIB_DIR (needs the real librknnrt.so)")
    endif()
    add_executable(benchmark ocr/benchmark.cpp)
    target_link_libraries(benchmark PRIVATE ocr_core)
    target_link_options(benchmark PRIVATE -Wl,--allow-shlib-undefined)
endif()
