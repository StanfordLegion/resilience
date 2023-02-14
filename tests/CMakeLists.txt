list(APPEND RESILIENCE_TESTS
  # futures
  # futures_2
  futures_multiple_checkpoints
  # index_tasks
  # index_tasks_2d
  # index_tasks_3d
  # partition_equal_1d
  # partition_equal_1d_2
  # partition_equal_1d_3
  # partition_equal_2d
  # partition_equal_3d
  # partition_field
  # partition_image
  # region
)

foreach(RESILIENCE_TEST ${RESILIENCE_TESTS})
  add_executable(${RESILIENCE_TEST} ${RESILIENCE_TEST}.cc)
  target_link_libraries(${RESILIENCE_TEST} PRIVATE ResilienceStatic)
  set_property(TARGET ${RESILIENCE_TEST} PROPERTY CXX_STANDARD 11)
endforeach()

