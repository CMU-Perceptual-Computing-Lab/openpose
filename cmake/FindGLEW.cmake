
# GLEW_FOUND
# GLEW_INCLUDE_DIR
# GLEW_LIBRARY

# GLEW_BINARY (win32 only)


find_path(GLEW_INCLUDE_DIR GL/glew.h

    PATHS
    $ENV{GLEW_DIR}
    /usr
    /usr/local
    /sw
    /opt/local

    PATH_SUFFIXES
    /include

    DOC "The directory where GL/glew.h resides")

if (X64)
    set(GLEW_BUILD_DIR Release/x64)
else()
    set(GLEW_BUILD_DIR Release/Win32)
endif()

find_library(GLEW_LIBRARY NAMES GLEW glew glew32 glew32s

    PATHS
    $ENV{GLEW_DIR}
    /usr
    /usr/local
    /sw
    /opt/local

    # authors prefered choice for development
    /build
    /build-release
    /build-debug
    $ENV{GLEW_DIR}/build
    $ENV{GLEW_DIR}/build-release
    $ENV{GLEW_DIR}/build-debug

    PATH_SUFFIXES
    /lib
    /lib64
    /lib/${GLEW_BUILD_DIR}

    DOC "The GLEW library")

if(WIN32)

    find_file(GLEW_BINARY NAMES glew32.dll glew32s.dll

        HINTS
        ${GLEW_INCLUDE_DIR}/..

        PATHS
        $ENV{GLEW_DIR}

        PATH_SUFFIXES
        /bin
        /bin/${GLEW_BUILD_DIR}

        DOC "The GLEW binary")

endif()
    
find_package_handle_standard_args(GLEW REQUIRED_VARS GLEW_INCLUDE_DIR GLEW_LIBRARY)
mark_as_advanced(GLEW_INCLUDE_DIR GLEW_LIBRARY)
