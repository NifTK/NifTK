project(uk_ac_ucl_cmic_newvisualization)

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE NEWVISUALIZATION_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDS MitkCore MitkQtWidgets MitkQtWidgetsExt niftkCore niftkCoreGui
)
