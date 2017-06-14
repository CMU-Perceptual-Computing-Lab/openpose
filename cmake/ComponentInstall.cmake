
# Execute cmake_install.cmake wrapper that allows to pass both DESTDIR and COMPONENT environment variable

execute_process(
    COMMAND ${CMAKE_COMMAND} -DCOMPONENT=$ENV{COMPONENT} -P cmake_install.cmake
)
