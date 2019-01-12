set(PYTHON_FILES
    openpose.py
    __init__.py
    _openpose.cpp)

pybind11_add_module(openpose_python _openpose.cpp)

target_link_libraries(openpose_python PRIVATE pybind11::module openpose ${OpenPose_3rdparty_libraries})
SET_TARGET_PROPERTIES(openpose_python PROPERTIES PREFIX "")
configure_file(__init__.py __init__.py)

# install(TARGETS openpose_python DESTINATION python)
# install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ DESTINATION python/openpose FILES_MATCHING PATTERN "*.so")
# install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ DESTINATION python/openpose FILES_MATCHING PATTERN "*.py")
