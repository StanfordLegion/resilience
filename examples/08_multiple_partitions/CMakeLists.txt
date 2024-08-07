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

add_executable(multiple_partitions multiple_partitions.cc)
target_link_libraries(multiple_partitions Legion::Legion ResilienceStatic)
set_target_properties(multiple_partitions PROPERTIES CXX_STANDARD 17)
add_test(NAME multiple_partitions COMMAND ${PROJECT_SOURCE_DIR}/tests/runner.py -- ${RESILIENCE_TEST_LAUNCHER} $<TARGET_FILE:multiple_partitions> -lg:safe_ctrlrepl 2)
add_test(NAME multiple_partitions_disable COMMAND ${PROJECT_SOURCE_DIR}/tests/runner.py -- ${RESILIENCE_TEST_LAUNCHER} $<TARGET_FILE:multiple_partitions> -lg:safe_ctrlrepl 2 -checkpoint:disable)
