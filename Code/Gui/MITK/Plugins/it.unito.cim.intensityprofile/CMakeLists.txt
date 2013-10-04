project(it_unito_cim_intensityprofile)

# FIXME
# Temporary workaround for CTK bug of not exposing external project library dirs.
# Should be removed as soon as this is fixed in CTK. (espakm)
if(EXISTS "${CTK_DIR}/qRestAPI-build/qRestAPIConfig.cmake")
  include(${CTK_DIR}/qRestAPI-build/qRestAPIConfig.cmake)
  link_directories(${qRestAPI_LIBRARY_DIRS})
endif()
if(EXISTS "${CTK_DIR}/QuaZip-build/QuaZipConfig.cmake")
  include(${CTK_DIR}/QuaZip-build/QuaZipConfig.cmake)
  link_directories(${QUAZIP_LIBRARY_DIRS})
endif()

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE CIM_DCE_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES MitkExt ImageStatistics Qmitk QmitkExt niftkCore
)
