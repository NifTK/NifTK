PROJECT(uk_ac_ucl_cmic_xnat)

# These should be included automatically but it is not.
#include(${CMAKE_SOURCE_DIR}/CMake/PackageDepends/MITK_curl_Config.cmake)
#include(${CMAKE_SOURCE_DIR}/CMake/PackageDepends/MITK_zlib_Config.cmake)

set(QT_USE_QTNETWORK TRUE)

# To process the result of the queries in XML format 
#set(QT_USE_QTXML TRUE)

# To process the result of the queries in json format 
set(QT_USE_QTSCRIPT TRUE)

message(STATUS "current include dir: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "current source dir: ${CMAKE_CURRENT_SOURCE_DIR}/src/internal")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/internal)

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE XNAT_EXPORTS
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES Mitk MitkExt Qmitk QmitkExt XnatRestWidgets CTK
)
