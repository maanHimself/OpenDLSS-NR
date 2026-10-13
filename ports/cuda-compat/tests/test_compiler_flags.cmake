file(MAKE_DIRECTORY "${NR_TEST_DIR}/safe-source")
file(WRITE "${NR_TEST_DIR}/safe-source/CMakeLists.txt"
  "cmake_minimum_required(VERSION 3.24)\ninclude(\"${NR_SOURCE_DIR}/cmake/CheckArithmeticFlags.cmake\")\nnr_check_arithmetic_flags()\nproject(flag_test LANGUAGES NONE)\n")
set(case 0)
foreach(flag --use_fast_math -use_fast_math --ftz=true -ftz=true "--ftz true" "-ftz true"
             --prec-div=false -prec-div=false "--prec-div false" "-prec-div false"
             --prec-sqrt=false -prec-sqrt=false "--prec-sqrt false" "-prec-sqrt false"
             -ffast-math /fp:fast -Xcompiler=-ffast-math --compiler-options=-ffast-math
             -Xcompiler=-fPIC,-ffast-math "-Xcompiler -fPIC,-ffast-math"
             "--compiler-options -fPIC,-ffast-math" -Ofast -Xcompiler=-Ofast
             "-Xcompiler -fPIC,-Ofast" --options-file=flags.txt -optf=flags.txt
             "--options-file flags.txt" "-optf flags.txt" @flags.txt -Xcompiler=@flags.txt)
  foreach(variable CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG
                   CMAKE_CUDA_COMPILER_ARG1 CMAKE_CXX_COMPILER_ARG1)
    math(EXPR case "${case}+1")
    execute_process(COMMAND "${CMAKE_COMMAND}" --fresh -S "${NR_SOURCE_DIR}" -B "${NR_TEST_DIR}/bad-${case}"
      "-D${variable}=${flag}" RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(status EQUAL 0 OR NOT "${output}${error}" MATCHES "unsafe flags")
      message(FATAL_ERROR "Guard did not reject ${variable}=${flag}: ${output}${error}")
    endif()
  endforeach()
endforeach()
foreach(flag --ftz=false -ftz=false "--ftz false" --prec-div=true -prec-div=true
             "--prec-div true" --prec-sqrt=true -prec-sqrt=true "--prec-sqrt true"
             -lineinfo "-DNAME=--use_fast_math" "-I/path/--ftz=true" --use_fast_math_suffix
             "\"-I/path with --ftz=true\"" "\"-DNAME=--prec-div false\""
             "-Xcompiler -fPIC,-fno-fast-math" --compiler-options=-fPIC,-fno-fast-math)
  math(EXPR case "${case}+1")
  execute_process(COMMAND "${CMAKE_COMMAND}" --fresh -S "${NR_TEST_DIR}/safe-source" -B "${NR_TEST_DIR}/safe-${case}"
    "-DCMAKE_CUDA_FLAGS=${flag}" RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Guard rejected safe/unrelated flag ${flag}: ${output}${error}")
  endif()
endforeach()
foreach(environment "CXXFLAGS=-ffast-math" "CUDAFLAGS=-use_fast_math"
                    "NVCC_PREPEND_FLAGS=-use_fast_math" "NVCC_APPEND_FLAGS=--prec-div=false"
                    "NVCC_APPEND_FLAGS=-Xcompiler=-Ofast" "CUDACXX=nvcc -Xcompiler=-Ofast"
                    "CXX=c++ -Ofast" "NVCC_APPEND_FLAGS=--options-file flags.txt")
  math(EXPR case "${case}+1")
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env "${environment}" "${CMAKE_COMMAND}"
    --fresh -S "${NR_SOURCE_DIR}" -B "${NR_TEST_DIR}/environment-${case}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(status EQUAL 0 OR NOT "${output}${error}" MATCHES "unsafe flags")
    message(FATAL_ERROR "Guard did not reject ${environment}: ${output}${error}")
  endif()
endforeach()
foreach(selection "-DCMAKE_CUDA_HOST_COMPILER=${CMAKE_COMMAND}" "-DCMAKE_CUDA_FLAGS=-ccbin=${CMAKE_COMMAND}")
  math(EXPR case "${case}+1")
  execute_process(COMMAND "${CMAKE_COMMAND}" --fresh -S "${NR_SOURCE_DIR}" -B "${NR_TEST_DIR}/host-${case}"
    "${selection}" RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(status EQUAL 0 OR NOT "${output}${error}" MATCHES "host compiler")
    message(FATAL_ERROR "Mismatched/overridden host accepted: ${output}${error}")
  endif()
endforeach()
execute_process(COMMAND "${CMAKE_COMMAND}" --fresh -S "${NR_SOURCE_DIR}" -B "${NR_TEST_DIR}/host-suffix"
  -DCMAKE_CXX_COMPILER_ARG1=-v RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(status EQUAL 0 OR NOT "${output}${error}" MATCHES "command suffixes are unsupported")
  message(FATAL_ERROR "Dropped host suffix accepted: ${output}${error}")
endif()
math(EXPR case "${case}+1")
execute_process(COMMAND "${CMAKE_COMMAND}" -E env --unset=NVCC_PREPEND_FLAGS --unset=NVCC_APPEND_FLAGS
  "${CMAKE_COMMAND}" --fresh -S "${NR_SOURCE_DIR}" -B "${NR_TEST_DIR}/build-env"
  -DBUILD_TESTING=OFF RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Build environment test configure failed: ${output}${error}")
endif()
foreach(environment "NVCC_PREPEND_FLAGS=-use_fast_math" "NVCC_APPEND_FLAGS=-Ofast"
                    "NVCC_APPEND_FLAGS=--prec-div=true" "NVCC_APPEND_FLAGS=-optf flags.txt")
  math(EXPR case "${case}+1")
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env --unset=NVCC_PREPEND_FLAGS --unset=NVCC_APPEND_FLAGS
    "${environment}" "${CMAKE_COMMAND}"
    --build "${NR_TEST_DIR}/build-env" --target nr_cuda
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(status EQUAL 0 OR NOT "${output}${error}" MATCHES "unsafe flags|changed; reconfigure")
    message(FATAL_ERROR "Build accepted injected environment: ${output}${error}")
  endif()
endforeach()
message(STATUS "Compiler flag configure/build cases passed: ${case}")
