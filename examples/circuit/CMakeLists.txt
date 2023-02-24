#------------------------------------------------------------------------------#
# Copyright 2023 Kitware, Inc., Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------#

set(CPU_SOURCES
  circuit.h         circuit.cc
                    circuit_cpu.cc
                    circuit_init.cc
  circuit_mapper.h  circuit_mapper.cc
)
if(Legion_USE_CUDA)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)
  set(GPU_SOURCES circuit_gpu.cu)
  cuda_add_executable(circuit ${CPU_SOURCES} ${GPU_SOURCES})
elseif(Legion_USE_HIP)
  set(GPU_SOURCES circuit_gpu.cu)
  if(Legion_HIP_TARGET STREQUAL "CUDA")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)
    cuda_add_executable(circuit ${CPU_SOURCES} ${GPU_SOURCES})
    target_compile_definitions(circuit PRIVATE __HIP_PLATFORM_NVIDIA__)
  elseif (Legion_HIP_TARGET STREQUAL "ROCM")
    hip_add_executable(circuit ${CPU_SOURCES} ${GPU_SOURCES})
    target_compile_definitions(circuit PRIVATE __HIP_PLATFORM_AMD__)
  endif()
else()
  add_executable(circuit ${CPU_SOURCES})
endif()
target_link_libraries(circuit Legion::Legion ResilienceStatic)
set_target_properties(circuit PROPERTIES CXX_STANDARD 11)
add_test(NAME circuit COMMAND ${PROJECT_SOURCE_DIR}/tests/runner.py -- $<TARGET_FILE:circuit>)
