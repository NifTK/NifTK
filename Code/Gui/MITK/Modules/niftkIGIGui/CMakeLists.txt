MITK_CREATE_MODULE(niftkIGIGui
  EXPORT_DEFINE NIFTKIGIGUI_EXPORT
  INCLUDE_DIRS ToolsGui Common
  DEPENDS Mitk MitkExt Qmitk QmitkExt MitkIGT MitkIGTUI niftkMitkExt niftkIGI
  PACKAGE_DEPENDS QT QVTK CTK
  QT_MODULE 
)

TARGET_LINK_LIBRARIES(niftkIGIGui
                      igtlutil
                      niftylink
                      OpenIGTLink
)
