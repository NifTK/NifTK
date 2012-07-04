project(uk_ac_ucl_cmic_mitksegmentation)

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE MITKSEGMENTATION_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES ClippingTools Segmentation QmitkExt niftkMitkExt niftkQmitkExt
)
