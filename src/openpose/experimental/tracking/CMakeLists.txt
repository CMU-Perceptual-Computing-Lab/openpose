set(SOURCES_OP_TRACKING
    defineTemplates.cpp
    personIdExtractor.cpp
    personTracker.cpp
    pyramidalLK.cpp
    pyramidalLK.cu)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_TRACKING_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_TRACKING})
set(SOURCES_OP_TRACKING_WITH_CP ${SOURCES_OP_TRACKING_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_TRACKING_WITH_CP} PARENT_SCOPE)

if (UNIX AND NOT APPLE)
	add_library(openpose_tracking ${SOURCES_OP_TRACKING})

	set_target_properties(openpose_tracking PROPERTIES COMPILE_FLAGS ${OP_CXX_FLAGS})

	install(TARGETS openpose_tracking
	    EXPORT OpenPose
	    RUNTIME DESTINATION bin
	    LIBRARY DESTINATION lib
	    ARCHIVE DESTINATION lib/openpose)
endif (UNIX AND NOT APPLE)
