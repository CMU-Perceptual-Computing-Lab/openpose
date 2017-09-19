
# 
# Find LaTeX
# 

find_package(LATEX)
if(NOT LATEX_FOUND)
    message(STATUS "Disabled generation of documentation (missing LaTeX).")
    return()
endif()


# 
# Target name
# 

set(target docs-manual)
message(STATUS "Doc ${target}")


# 
# Input and output files
# 

set(source "${CMAKE_CURRENT_SOURCE_DIR}/openpose.tex")
set(pdf    "${CMAKE_CURRENT_BINARY_DIR}/openpose.pdf")


# 
# Create documentation
# 

# Invoke LaTeX
add_custom_command(
    OUTPUT            ${pdf}
    DEPENDS           ${source}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND           ${PDFLATEX_COMPILER} \"${source}\"
    COMMAND           ${PDFLATEX_COMPILER} \"${source}\"
    COMMAND           ${PDFLATEX_COMPILER} \"${source}\"
    COMMENT           "Creating LaTeX documentation."
)

# Declare target
add_custom_target(${target} ALL DEPENDS ${pdf})
add_dependencies(docs ${target})


# 
# Deployment
# 

# PDF file
install(FILES ${pdf}
    DESTINATION "${INSTALL_DOC}"
    COMPONENT docs
)
