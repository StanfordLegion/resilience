list(APPEND RESILIENCE_TESTS
  futures
  futures_2
  futures_multiple_checkpoints
  futures_repeat
  futures_repeat_2
  futures_repeat_3
  futures_repeat_4
  index_tasks
  index_tasks_2d
  index_tasks_3d
  partition_equal_1d
  partition_equal_1d_2
  partition_equal_1d_3
  partition_equal_2d
  partition_equal_3d
  partition_field
  partition_image
  partition_preimage
  region
  region_destroy
)

foreach(RESILIENCE_TEST ${RESILIENCE_TESTS})
  add_executable(${RESILIENCE_TEST} ${RESILIENCE_TEST}.cc)
  target_link_libraries(${RESILIENCE_TEST} ResilienceStatic)
  set_target_properties(${RESILIENCE_TEST} PROPERTIES CXX_STANDARD 17)
  add_test(NAME ${RESILIENCE_TEST} COMMAND ${PROJECT_SOURCE_DIR}/tests/runner.py -- ${RESILIENCE_TEST_LAUNCHER} $<TARGET_FILE:${RESILIENCE_TEST}> -lg:safe_ctrlrepl 2)
  add_test(NAME ${RESILIENCE_TEST}_disable COMMAND ${PROJECT_SOURCE_DIR}/tests/runner.py -- ${RESILIENCE_TEST_LAUNCHER} $<TARGET_FILE:${RESILIENCE_TEST}> -lg:safe_ctrlrepl 2 -checkpoint:disable)
endforeach()

