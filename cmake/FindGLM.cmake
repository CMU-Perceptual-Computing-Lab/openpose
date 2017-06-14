
# GLM_FOUND
# GLM_INCLUDE_DIR

include(FindPackageHandleStandardArgs)

FIND_PATH(GLM_INCLUDE_DIR glm/glm.hpp

    PATHS
    $ENV{GLM_DIR}
    /usr
    /usr/local
    /sw
    /opt/local

	PATH_SUFFIXES
    /include

    DOC "The directory where glm/glm.hpp resides.")
    
find_package_handle_standard_args(GLM REQUIRED_VARS GLM_INCLUDE_DIR)

mark_as_advanced(GLM_INCLUDE_DIR)
