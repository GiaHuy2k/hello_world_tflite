cmake_minimum_required(VERSION 3.5)

set(tflite_dir "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/lite")
set(signal_dir "${CMAKE_CURRENT_SOURCE_DIR}/signal")
set(tfmicro_dir "${tflite_dir}/micro")
set(tfmicro_frontend_dir "${tflite_dir}/experimental/microfrontend/lib")
set(tfmicro_kernels_dir "${tfmicro_dir}/kernels")

file(GLOB tflite
          "${tflite_dir}/*.c"
          "${tflite_dir}/*.cc")

file(GLOB srcs_micro
          "${tfmicro_dir}/*.cc"
          "${tfmicro_dir}/*.c")

file(GLOB src_micro_frontend
          "${tfmicro_frontend_dir}/*.c"
          "${tfmicro_frontend_dir}/*.cc")

file(GLOB srcs_tflite_bridge
          "${tfmicro_dir}/tflite_bridge/*.c"
          "${tfmicro_dir}/tflite_bridge/*.cc")

file(GLOB srcs_kernels
          "${tfmicro_kernels_dir}/*.c"
          "${tfmicro_kernels_dir}/*.cc")

set(signal_srcs
          "${signal_dir}/micro/kernels/rfft.cc"
          "${signal_dir}/micro/kernels/window.cc"
          "${signal_dir}/src/kiss_fft_wrappers/kiss_fft_float.cc"
          "${signal_dir}/src/kiss_fft_wrappers/kiss_fft_int16.cc"
          "${signal_dir}/src/kiss_fft_wrappers/kiss_fft_int32.cc"
          "${signal_dir}/src/rfft_float.cc"
          "${signal_dir}/src/rfft_int16.cc"
          "${signal_dir}/src/rfft_int32.cc"
          "${signal_dir}/src/window.cc")

set(lib_srcs
        "${srcs_micro}"
        "${srcs_kernels}"
        "${srcs_tflite_bridge}"
        "${esp_nn_kernels}"
        "${src_micro_frontend}"
        "${signal_srcs}"
        "${tflite}"
        "${tflite_dir}/kernels/kernel_util.cc"
        "${tflite_dir}/micro/memory_planner/greedy_memory_planner.cc"
        "${tflite_dir}/micro/memory_planner/linear_memory_planner.cc"
        "${tflite_dir}/micro/arena_allocator/non_persistent_arena_buffer_allocator.cc"
        "${tflite_dir}/micro/arena_allocator/persistent_arena_buffer_allocator.cc"
        "${tflite_dir}/micro/arena_allocator/recording_single_arena_buffer_allocator.cc"
        "${tflite_dir}/micro/arena_allocator/single_arena_buffer_allocator.cc"
        "${tflite_dir}/core/c/common.cc"
        "${tflite_dir}/core/api/error_reporter.cc"
        "${tflite_dir}/core/api/flatbuffer_conversions.cc"
        "${tflite_dir}/core/api/tensor_utils.cc"
        "${tflite_dir}/kernels/internal/common.cc"
        "${tflite_dir}/kernels/internal/quantization_util.cc"
        "${tflite_dir}/kernels/internal/portable_tensor_utils.cc"
        "${tflite_dir}/kernels/internal/tensor_utils.cc"
        "${tflite_dir}/kernels/internal/tensor_ctypes.cc"
        "${tflite_dir}/kernels/internal/reference/portable_tensor_utils.cc"
        "${tflite_dir}/kernels/internal/reference/comparisons.cc"
        "${tflite_dir}/schema/schema_utils.cc")

idf_component_register(
            SRCS "${lib_srcs}"
            INCLUDE_DIRS "." "third_party/gemmlowp"
                         "third_party/flatbuffers/include"
                         "third_party/ruy"
                         "third_party/kissfft"
                         "signal/micro/kernels"
                         "signal/src"
                         "signal/src/kiss_fft_wrappers"
            REQUIRES ${pub_req}
            PRIV_REQUIRES ${priv_req})

# Reduce the level of paranoia to be able to compile TF sources
target_compile_options(${COMPONENT_LIB} PRIVATE
  -Wno-error=attributes
  -Wno-maybe-uninitialized
  -Wno-missing-field-initializers
  -Wno-error=sign-compare
  -Wno-error=double-promotion
  -Wno-type-limits)

if("${IDF_VERSION_MAJOR}.${IDF_VERSION_MINOR}" VERSION_GREATER_EQUAL "5.0")
  target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=stringop-overread)
endif()

