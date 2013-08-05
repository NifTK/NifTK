project(it_unito_cim_intensityprofile)

# Temporary workaround for CTK bug of not exposing external project library dirs.
# Should be removed as soon as this is fixed in CTK. (espakm)
include(${CTK_DIR}/qRestAPI-build/qRestAPIConfig.cmake)
link_directories(${qRestAPI_LIBRARY_DIRS})

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE CIM_DCE_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES MitkExt ImageStatistics Qmitk QmitkExt niftkCore
)
