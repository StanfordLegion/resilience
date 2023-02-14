list(APPEND RESILIENCE_SRC
  resilience.cc resilience.h
)

add_library(ResilienceObjectFiles OBJECT ${RESILIENCE_SRC})

target_include_directories(ResilienceObjectFiles
  PUBLIC
    ${PROJECT_SOURCE_DIR}/vendor/cereal/include
)

target_link_libraries(ResilienceObjectFiles
  PUBLIC
    Legion::Legion
    Legion::Realm
)

set_property(TARGET ResilienceObjectFiles PROPERTY CXX_STANDARD 11)

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
