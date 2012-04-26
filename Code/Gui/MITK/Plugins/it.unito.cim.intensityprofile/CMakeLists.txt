project(it_unito_cim_intensityprofile)

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE CIM_DCE_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES MitkExt ImageStatistics Qmitk QmitkExt
)
