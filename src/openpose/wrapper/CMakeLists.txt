set(SOURCES
  defineTemplates.cpp
  wrapperStructFace.cpp
  wrapperStructHand.cpp
  wrapperStructInput.cpp
  wrapperStructOutput.cpp
  wrapperStructPose.cpp)

add_library(wrapper SHARED ${SOURCES})
target_link_libraries(wrapper thread pose hand core face filestream gui producer utilities)
