
# These should be included automatically but it is not.
#include(${CMAKE_SOURCE_DIR}/CMake/PackageDepends/MITK_curl_Config.cmake)
#include(${CMAKE_SOURCE_DIR}/CMake/PackageDepends/MITK_zlib_Config.cmake)

MITK_CREATE_MODULE(XnatRestWidgets
  EXPORT_DEFINE XnatRestWidgets_EXPORT
  DEPENDS XnatRest
  PACKAGE_DEPENDS curl zlib
  QT_MODULE
)

if(CMAKE_COMPILER_IS_GNUCXX)
  set_target_properties(XnatRestWidgets PROPERTIES COMPILE_FLAGS "-fvisibility=default")
endif()
