
# Findcppcheck results:
# cppcheck_FOUND
# cppcheck_EXECUTABLE

include(FindPackageHandleStandardArgs)

# work around CMP0053, see http://public.kitware.com/pipermail/cmake/2014-November/059117.html
set(PROGRAMFILES_x86_ENV "PROGRAMFILES(x86)")

find_program(cppcheck_EXECUTABLE
    NAMES
        cppcheck
    PATHS
        "${CPPCHECK_DIR}"
        "$ENV{CPPCHECK_DIR}"
        "$ENV{PROGRAMFILES}/Cppcheck"
        "$ENV{${PROGRAMFILES_x86_ENV}}/Cppcheck"
)

find_package_handle_standard_args(cppcheck
	FOUND_VAR
        cppcheck_FOUND
    REQUIRED_VARS
        cppcheck_EXECUTABLE
)

mark_as_advanced(cppcheck_EXECUTABLE)
