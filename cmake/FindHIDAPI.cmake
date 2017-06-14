
# HIDAPI_FOUND
# HIDAPI_INCLUDE_DIRS
# HIDAPI_LIBRARIES

find_path(HIDAPI_INCLUDE_DIRS
	NAMES hidapi/hidapi.h
	/usr/include
	/usr/local/include
	/sw/include
	/opt/local/include
	DOC "The directory where hidapi/hidapi.h resides")

find_library(HIDAPI_LIBRARIES
	NAMES hidapi-hidraw hidapi-libusb
	PATHS
	/usr/lib64
	/usr/local/lib64
	/sw/lib64
	/opt/loca/lib64
	/usr/lib
	/usr/local/lib
	/sw/lib
	/opt/local/lib
	DOC "The hidapi library")


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HIDAPI REQUIRED_VARS HIDAPI_LIBRARIES HIDAPI_INCLUDE_DIRS)

mark_as_advanced(HIDAPI_INCLUDE_DIR HIDAPI_LIBRARY)

