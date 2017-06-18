
# 
# Configuration for all sub-projects
# 

# Generate version-header
configure_file(version.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/${META_PROJECT_NAME}/${META_PROJECT_NAME}-version.h)


# 
# Sub-projects
# 

# Libraries
set(IDE_FOLDER "")
add_subdirectory(core)
add_subdirectory(face)
add_subdirectory(filestream)
add_subdirectory(gui)
add_subdirectory(hand)
add_subdirectory(pose)
add_subdirectory(producer)
add_subdirectory(thread)
add_subdirectory(utilities)
add_subdirectory(wrapper)

# 
# Deployment
# 

# Deploy generated headers
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/${META_PROJECT_NAME} DESTINATION include COMPONENT dev)
