
# 
# Target 'docs'
# 

if(NOT OPTION_BUILD_DOCS)
    return()
endif()

add_custom_target(docs)


# 
# Documentation
# 

add_subdirectory(api-docs)
add_subdirectory(manual)
