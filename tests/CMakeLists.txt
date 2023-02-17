list(APPEND RESILIENCE_TESTS
  futures
  futures_2
  futures_multiple_checkpoints
  index_tasks
  index_tasks_2d
  index_tasks_3d
  partition_equal_1d
  partition_equal_1d_2
  partition_equal_1d_3
  partition_equal_2d
  partition_equal_3d
  # partition_field
  # partition_image
  region
  region_destroy
)

list(APPEND TESTS_FAIL_REPLAY
  index_tasks
  partition_equal_1d
  partition_equal_1d_2
  partition_equal_1d_3
  partition_equal_2d
  partition_equal_3d
)

foreach(RESILIENCE_TEST ${RESILIENCE_TESTS})
  add_executable(${RESILIENCE_TEST} ${RESILIENCE_TEST}.cc)
  target_link_libraries(${RESILIENCE_TEST} ResilienceStatic)
  set_target_properties(${RESILIENCE_TEST} PROPERTIES CXX_STANDARD 11)
  if(RESILIENCE_TEST IN_LIST TESTS_FAIL_REPLAY)
    add_test(NAME ${RESILIENCE_TEST} COMMAND $<TARGET_FILE:${RESILIENCE_TEST}>)
  else()
    add_test(NAME ${RESILIENCE_TEST} COMMAND ${PROJECT_SOURCE_DIR}/tests/runner.sh $<TARGET_FILE:${RESILIENCE_TEST}>)
  endif()
endforeach()

