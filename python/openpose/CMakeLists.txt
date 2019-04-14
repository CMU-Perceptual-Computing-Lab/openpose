set(PYTHON_FILES
    openpose.py
    __init__.py
    openpose_python.cpp)

pybind11_add_module(pyopenpose openpose_python.cpp)

target_link_libraries(pyopenpose PRIVATE pybind11::module openpose ${OpenPose_3rdparty_libraries})
SET_TARGET_PROPERTIES(pyopenpose PROPERTIES PREFIX "")
configure_file(__init__.py __init__.py)

install(TARGETS pyopenpose DESTINATION python)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ DESTINATION python/openpose FILES_MATCHING PATTERN "*.so")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ DESTINATION python/openpose FILES_MATCHING PATTERN "*.py")
