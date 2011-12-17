PROJECT(uk_ac_ucl_cmic_midasgeneralsegmentor)

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE MIDASGENERALSEGMENTOR_EXPORTS
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES CTK QmitkExt niftkMitkExt niftkQmitkExt
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} 
                      niftkQt
                      niftkITK
)
