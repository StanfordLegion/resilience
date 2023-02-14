list(APPEND RESILIENCE_SRC
  resilience.cc resilience.h
)

add_library(ResilienceObjectFiles OBJECT ${RESILIENCE_SRC})

add_library(ResilienceStatic
  STATIC
    $<TARGET_OBJECTS:ResilienceObjectFiles>
)

add_library(ResilienceShared
  SHARED
    $<TARGET_OBJECTS:ResilienceObjectFiles>
)
