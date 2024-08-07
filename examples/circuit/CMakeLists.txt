#------------------------------------------------------------------------------#
# Copyright 2024 Kitware, Inc., Stanford University
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
  set(GPU_SOURCES circuit_gpu.cu)
  add_executable(circuit ${CPU_SOURCES} ${GPU_SOURCES})
  set_target_cuda_standard(circuit STANDARD ${Legion_CUDA_STANDARD})
  set_target_cuda_architectures(circuit ARCHITECTURES ${Legion_CUDA_ARCH})
  set_target_cuda_warnings_and_errors(circuit WARN_AS_ERROR ${Legion_BUILD_WARN_AS_ERROR})
elseif(Legion_USE_HIP)
  set(GPU_SOURCES circuit_gpu.cu)
  if(Legion_HIP_TARGET STREQUAL "CUDA")
    add_executable(circuit ${CPU_SOURCES} ${GPU_SOURCES})
    target_compile_definitions(circuit PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories(circuit PRIVATE ${HIP_ROOT_DIR}/include)
  elseif (Legion_HIP_TARGET STREQUAL "ROCM")
    hip_add_executable(circuit ${CPU_SOURCES} ${GPU_SOURCES})
    target_compile_definitions(circuit PRIVATE __HIP_PLATFORM_AMD__)
  endif()
else()
  add_executable(circuit ${CPU_SOURCES})
endif()
target_link_libraries(circuit Legion::Legion ResilienceStatic)
set_target_properties(circuit PROPERTIES CXX_STANDARD 17)
add_test(NAME circuit COMMAND ${PROJECT_SOURCE_DIR}/tests/runner.py -- ${RESILIENCE_TEST_LAUNCHER} $<TARGET_FILE:circuit> -lg:eager_alloc_percentage 20 -lg:safe_ctrlrepl 2)
add_test(NAME circuit_disable COMMAND ${PROJECT_SOURCE_DIR}/tests/runner.py -- ${RESILIENCE_TEST_LAUNCHER} $<TARGET_FILE:circuit> -lg:eager_alloc_percentage 20 -lg:safe_ctrlrepl 2 -checkpoint:disable)
