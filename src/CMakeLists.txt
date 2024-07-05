list(APPEND RESILIENCE_SRC
  resilience.h resilience/resilience.inl resilience/resilience.cc
  resilience/resilience_c.h resilience/resilience_c.cc
  resilience/resilience_c_util.h
  resilience/future.h resilience/future.inl resilience/future.cc
  resilience/launcher.h resilience/launcher.inl
  resilience/mapper.h resilience/mapper.cc
  resilience/projection.h resilience/projection.cc
  resilience/serializer.h resilience/serializer.inl resilience/serializer.cc
  resilience/types.h
)

add_library(ResilienceObjectFiles OBJECT ${RESILIENCE_SRC})

target_include_directories(ResilienceObjectFiles
  PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/vendor/cereal/include
)

target_link_libraries(ResilienceObjectFiles
  PUBLIC
    Legion::Legion
    Legion::Realm
)

set_target_properties(ResilienceObjectFiles
  PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)

add_library(ResilienceStatic
  STATIC
    $<TARGET_OBJECTS:ResilienceObjectFiles>
)
set_target_properties(ResilienceStatic PROPERTIES OUTPUT_NAME legion_resilience_s)

target_link_libraries(ResilienceStatic
  PUBLIC
    Legion::Legion
    Legion::Realm
)

target_include_directories(ResilienceStatic
  PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/vendor/cereal/include
)

add_library(ResilienceShared
  SHARED
    $<TARGET_OBJECTS:ResilienceObjectFiles>
)
set_target_properties(ResilienceShared PROPERTIES OUTPUT_NAME legion_resilience)

target_link_libraries(ResilienceShared
  PUBLIC
    Legion::Legion
    Legion::Realm
)

target_include_directories(ResilienceShared
  PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/vendor/cereal/include
)

install(
  TARGETS ResilienceStatic ResilienceShared
  EXPORT ResilienceExports
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
  FILES ${PROJECT_SOURCE_DIR}/src/resilience.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  FILES ${PROJECT_SOURCE_DIR}/src/resilience/future.h
  FILES ${PROJECT_SOURCE_DIR}/src/resilience/future.inl
  FILES ${PROJECT_SOURCE_DIR}/src/resilience/launcher.h
  FILES ${PROJECT_SOURCE_DIR}/src/resilience/launcher.inl
  FILES ${PROJECT_SOURCE_DIR}/src/resilience/resilience.inl
  FILES ${PROJECT_SOURCE_DIR}/src/resilience/serializer.h
  FILES ${PROJECT_SOURCE_DIR}/src/resilience/serializer.inl
  FILES ${PROJECT_SOURCE_DIR}/src/resilience/types.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/resilience
)
