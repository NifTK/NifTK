project(it_unito_cim_intensityprofile)

# Temporary workaround until the qRestAPI lib directory is not exposed properly to
# projects depending on CTK. Should be removed as soon as this is fixed in CTK.
link_directories(${CTK_DIR}/qRestAPI-build)

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE CIM_DCE_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES MitkExt ImageStatistics Qmitk QmitkExt niftkCore
)
