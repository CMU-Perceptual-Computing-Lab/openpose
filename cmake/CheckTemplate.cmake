
#
# Get cmake-init latest commit SHA on master
#

file(DOWNLOAD
    "https://api.github.com/repos/cginternals/cmake-init/commits/master"
    "${PROJECT_BINARY_DIR}/cmake-init.github.data"
)
file(READ
    "${PROJECT_BINARY_DIR}/cmake-init.github.data"
    CMAKE_INIT_INFO
)

string(REGEX MATCH
    "\"sha\": \"([0-9a-f]+)\","
    CMAKE_INIT_SHA
    ${CMAKE_INIT_INFO})

string(SUBSTRING
    ${CMAKE_INIT_SHA}
    8
    40
    CMAKE_INIT_SHA
)

#
# Get latest cmake-init commit on this repository
#

# APPLIED_CMAKE_INIT_SHA can be set by parent script
if(NOT APPLIED_CMAKE_INIT_SHA)
    # [TODO]: Get from git commit list (see cmake_init/source/scripts/check_template.sh)
    set(APPLIED_CMAKE_INIT_SHA "")
endif ()

if("${APPLIED_CMAKE_INIT_SHA}" STREQUAL "")
    message(WARNING
        "No cmake-init version detected, could not verify up-to-dateness. "
        "Set the cmake-init version by defining a META_CMAKE_INIT_SHA for your project."
    )
    return()
endif()

if(${APPLIED_CMAKE_INIT_SHA} STREQUAL ${CMAKE_INIT_SHA})
    message(STATUS "cmake-init template is up-to-date (${CMAKE_INIT_SHA})")
else()
    message(STATUS "cmake-init template needs an update https://github.com/cginternals/cmake-init/compare/${APPLIED_CMAKE_INIT_SHA}...master")
endif()
