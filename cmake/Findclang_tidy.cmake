
# Findclang_tidy results:
# clang_tidy_FOUND
# clang_tidy_EXECUTABLE

include(FindPackageHandleStandardArgs)

find_program(clang_tidy_EXECUTABLE
    NAMES
        clang-tidy-3.5
        clang-tidy-3.6
        clang-tidy-3.7
        clang-tidy-3.8
        clang-tidy-3.9
        clang-tidy-4.0
    PATHS
        "${CLANG_TIDY_DIR}"
)

find_package_handle_standard_args(clang_tidy
    FOUND_VAR
        clang_tidy_FOUND
    REQUIRED_VARS
        clang_tidy_EXECUTABLE
)

mark_as_advanced(clang_tidy_EXECUTABLE)