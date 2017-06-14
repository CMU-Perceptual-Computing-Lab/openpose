
# ASSIMP_FOUND
# ASSIMP_INCLUDE_DIR
# ASSIMP_LIBRARY_RELEASE
# ASSIMP_LIBRARY_DEBUG
# ASSIMP_LIBRARIES
# ASSIMP_BINARY (win32 only)

include(FindPackageHandleStandardArgs)

find_path(ASSIMP_INCLUDE_DIR assimp/Importer.hpp

    PATHS
    $ENV{ASSIMP_DIR}
    $ENV{PROGRAMFILES}/Assimp
    /usr
    /usr/local
    /sw
    /opt/local

    PATH_SUFFIXES
    /include

    DOC "The directory where assimp/Importer.hpp etc. resides")

if(MSVC AND X64)
    set(ASSIMP_PF "64")
else()
    set(ASSIMP_PF "86")
endif()

find_library(ASSIMP_LIBRARY_RELEASE NAMES assimp
    
    HINTS
    ${ASSIMP_INCLUDE_DIR}/..
    
    PATHS
    $ENV{ASSIMP_DIR}
    /usr
    /usr/local
    /sw
    /opt/local

    PATH_SUFFIXES
    /lib
    /lib${ASSIMP_PF}
    /build/code
    /build-debug/code

    DOC "The Assimp library (release)")

find_library(ASSIMP_LIBRARY_DEBUG NAMES assimpd
    
    HINTS
    ${ASSIMP_INCLUDE_DIR}/..

    PATHS
    $ENV{ASSIMP_DIR}
    /usr
    /usr/local
    /sw
    /opt/local

    PATH_SUFFIXES
    /lib
    /lib${ASSIMP_PF}
    /build/code
    /build-debug/code

    DOC "The Assimp library (debug)")

set(ASSIMP_LIBRARIES "")
if(ASSIMP_LIBRARY_RELEASE AND ASSIMP_LIBRARY_DEBUG)
    set(ASSIMP_LIBRARIES 
        optimized   ${ASSIMP_LIBRARY_RELEASE}
        debug       ${ASSIMP_LIBRARY_DEBUG})
elseif(ASSIMP_LIBRARY_RELEASE)
    set(ASSIMP_LIBRARIES ${ASSIMP_LIBRARY_RELEASE})
elseif(ASSIMP_LIBRARY_DEBUG)
    set(ASSIMP_LIBRARIES ${ASSIMP_LIBRARY_DEBUG})
endif()

if(WIN32)

    find_file(ASSIMP_BINARY NAMES assimp.dll "assimp${ASSIMP_PF}.dll"

        HINTS
        ${ASSIMP_INCLUDE_DIR}/..
        
        PATHS
        $ENV{ASSIMP_DIR}

        PATH_SUFFIXES
        /bin
        /bin${ASSIMP_PF}

        DOC "The Assimp binary")

endif()

find_package_handle_standard_args(ASSIMP DEFAULT_MSG ASSIMP_LIBRARIES ASSIMP_INCLUDE_DIR)
mark_as_advanced(ASSIMP_FOUND ASSIMP_INCLUDE_DIR ASSIMP_LIBRARIES)
