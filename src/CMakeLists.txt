list(APPEND RESILIENCE_SRC
  resilience.h resilience/resilience.cc
  resilience/types.h
  resilience/future.h resilience/future.inl resilience/future.cc
  resilience/serializer.h resilience/serializer.cc
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
    CXX_STANDARD 11
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
