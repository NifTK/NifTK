project(uk_ac_ucl_cmic_midaseditor)

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE MIDASEDITOR_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES CTK Mitk MitkExt Qmitk QmitkExt niftkQmitkExt niftkMitkExt
)
