project(uk_ac_ucl_cmic_surgicalguidance)

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE SURGICALGUIDANCE_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES QmitkExt MitkIGT MitkIGTUI
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} 
                      igtlutil
                      niftylink
                      OpenIGTLink
)
