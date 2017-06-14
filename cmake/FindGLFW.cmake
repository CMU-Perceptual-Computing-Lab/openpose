
# GLFW_FOUND
# GLFW_INCLUDE_DIR
# GLFW_LIBRARY_RELEASE
# GLFW_LIBRARY_DEBUG
# GLFW_LIBRARIES
# GLFW_BINARY (win32 only)

include(FindPackageHandleStandardArgs)


find_path(GLFW_INCLUDE_DIR GLFW/glfw3.h

    PATHS
    $ENV{GLFW_DIR}
    /usr
    /usr/local
    /usr/include/GL
    /sw
    /opt/local
    /opt/graphics/OpenGL
    /opt/graphics/OpenGL/contrib/libglfw

    PATH_SUFFIXES
    /include

    DOC "The directory where GLFW/glfw.h resides")


set(GLFW_LIB_SUFFIX "")
if(MSVC14)
    set(GLFW_LIB_SUFFIX "vc2015")
elseif(MSVS12)
    set(GLFW_LIB_SUFFIX "vc2013")
elseif(MSVC11)
    set(GLFW_LIB_SUFFIX "vc2012")
elseif(MSVC10)
    set(GLFW_LIB_SUFFIX "vc2010")
elseif(MINGW)
    if(X64)
        set(GLFW_LIB_SUFFIX "mingw-w64")
    else()
        set(GLFW_LIB_SUFFIX "mingw")
    endif()
endif()

set(GLFW_NAMES glfw3 glfw)
set(GLFW_DEBUG_NAMES glfw3d glfwd)
if(WIN32)
    option(GLFW_SHARED "Use shared GLFW library (DLL)" ON)
    if(GLFW_SHARED)
        set(GLFW_NAMES glfw3dll glfwdll)
        set(GLFW_DEBUG_NAMES glfw3ddll glfwddll)
    endif()
endif()

find_library(GLFW_LIBRARY_RELEASE NAMES ${GLFW_NAMES}

    HINTS
    ${GLFW_INCLUDE_DIR}/..

    PATHS
    $ENV{GLFW_DIR}
    /lib/x64
    /lib/cocoa
    /usr
    /usr/local
    /sw
    /opt/local

    # authors prefered choice for development
    /build
    /build-release
    $ENV{GLFW_DIR}/build
    $ENV{GLFW_DIR}/build-release

    PATH_SUFFIXES
    /lib
    /lib64
    /lib-${GLFW_LIB_SUFFIX}
    /src # for from-source builds

    DOC "The GLFW library")

find_library(GLFW_LIBRARY_DEBUG NAMES ${GLFW_DEBUG_NAMES}

    HINTS
    ${GLFW_INCLUDE_DIR}/..

    PATHS
    $ENV{GLFW_DIR}
    /lib/x64
    /lib/cocoa
    /usr
    /usr/local
    /sw
    /opt/local

    # authors prefered choice for development
    /build
    /build-debug
    $ENV{GLFW_DIR}/build
    $ENV{GLFW_DIR}/build-debug

    PATH_SUFFIXES
    /lib
    /lib64
    /src # for from-source builds

    DOC "The GLFW library")

set(GLFW_LIBRARIES "")
if(GLFW_LIBRARY_RELEASE AND GLFW_LIBRARY_DEBUG)
    set(GLFW_LIBRARIES 
        optimized   ${GLFW_LIBRARY_RELEASE}
        debug       ${GLFW_LIBRARY_DEBUG})
elseif(GLFW_LIBRARY_RELEASE)
    set(GLFW_LIBRARIES ${GLFW_LIBRARY_RELEASE})
elseif(GLFW_LIBRARY_DEBUG)
    set(GLFW_LIBRARIES ${GLFW_LIBRARY_DEBUG})
endif()

if(WIN32 AND GLFW_SHARED)

    find_file(GLFW_BINARY glfw3.dll

        HINTS
        ${GLFW_INCLUDE_DIR}/..

        PATHS
        $ENV{GLFW_DIR}
        /lib/x64
        /lib/cocoa

        PATH_SUFFIXES
        /lib
        /bin
        /lib-${GLFW_LIB_SUFFIX}

        DOC "The GLFW binary")

endif()

if(APPLE)
    set(GLFW_cocoa_LIBRARY "-framework Cocoa" CACHE STRING "Cocoa framework for OSX")
    set(GLFW_iokit_LIBRARY "-framework IOKit" CACHE STRING "IOKit framework for OSX")
    set(GLFW_corevideo_LIBRARY "-framework CoreVideo" CACHE STRING "CoreVideo framework for OSX")
endif()

# GLFW is required to link statically for now (no deploy specified)

find_package_handle_standard_args(GLFW DEFAULT_MSG GLFW_LIBRARIES GLFW_INCLUDE_DIR)
mark_as_advanced(GLFW_FOUND GLFW_INCLUDE_DIR GLFW_LIBRARIES)
