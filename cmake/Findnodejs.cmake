
# NODEJS_FOUND
# NODEJS_INCLUDE_DIRS
# NODEJS_INCLUDE_DIR
# NODEJS_LIBUV_INCLUDE_DIR

include(FindPackageHandleStandardArgs)

find_path(NODEJS_INCLUDE_DIR node.h
    $ENV{NODEJS_HOME}
    $ENV{NODEJSDIR}
    $ENV{NODEJS_HOME}/src
    $ENV{NODEJSDIR}/src
    /usr/include/nodejs/src
    /usr/local/include/nodejs/src
    /usr/include
    /usr/local/include
    /sw/include
    /usr/local/include/node
    /opt/local/include
    DOC "The directory where node.h resides.")

find_path(NODEJS_LIBUV_INCLUDE_DIR uv.h
    $ENV{NODEJS_HOME}
    $ENV{NODEJSDIR}
    $ENV{NODEJS_HOME}/src
    $ENV{NODEJSDIR}/src
    $ENV{NODEJS_HOME}/deps/uv/include
    $ENV{NODEJSDIR}/deps/uv/include
    /usr/include/nodejs/deps/uv/include
    /usr/local/include/nodejs/deps/uv/include
    /usr/include
    /usr/local/include
    /sw/include
    /opt/local/include
    /usr/local/include/node
    DOC "The directory where uv.h resides.")

find_path(NODEJS_LIBV8_INCLUDE_DIR v8.h
    $ENV{NODEJS_HOME}
    $ENV{NODEJSDIR}
    $ENV{NODEJS_HOME}/src
    $ENV{NODEJSDIR}/src
    $ENV{NODEJS_HOME}/deps/v8/include
    $ENV{NODEJSDIR}/deps/v8/include
    /usr/include/nodejs/deps/uv/include
    /usr/local/include/nodejs/deps/uv/include
    /usr/include
    /usr/local/include
    /sw/include
    /opt/local/include
    /usr/local/include/node
    DOC "The directory where v8.h resides.")

set(NODEJS_INCLUDE_DIRS ${NODEJS_INCLUDE_DIR} ${NODEJS_LIBUV_INCLUDE_DIR} ${NODEJS_LIBV8_INCLUDE_DIR})

find_package_handle_standard_args(NODEJS REQUIRED_VARS NODEJS_INCLUDE_DIRS)
mark_as_advanced(NODEJS_INCLUDE_DIRS)
