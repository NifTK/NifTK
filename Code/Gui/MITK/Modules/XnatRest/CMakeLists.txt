
# These should be included automatically but it is not.
include(${CMAKE_SOURCE_DIR}/CMake/PackageDepends/MITK_curl_Config.cmake)

MITK_CREATE_MODULE(XnatRest
#  PACKAGE_DEPENDS curl
)

if(CMAKE_COMPILER_IS_GNUCXX)
  set_target_properties(XnatRest PROPERTIES COMPILE_FLAGS "-fvisibility=default")
endif()

# add testing dir
add_subdirectory(Testing)
