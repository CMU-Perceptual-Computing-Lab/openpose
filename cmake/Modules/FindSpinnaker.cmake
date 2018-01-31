# Based on `FindCaffe.cmake`

unset(SPINNAKER_FOUND)
unset(SPINNAKER_INCLUDE_DIRS)
unset(SPINNAKER_LIB)

find_path(SPINNAKER_INCLUDE_DIRS NAMES
  Spinnaker.h
  HINTS
  /usr/include/spinnaker/
  /usr/local/include/spinnaker/)

find_library(SPINNAKER_LIB NAMES Spinnaker
    HINTS
    /usr/lib
    /usr/local/lib)

if (SPINNAKER_INCLUDE_DIRS AND SPINNAKER_LIB)
  set(SPINNAKER_FOUND 1)
endif (SPINNAKER_INCLUDE_DIRS AND SPINNAKER_LIB)
