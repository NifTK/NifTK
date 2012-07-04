
MITK_CREATE_MODULE(niftkQmitkExt
  EXPORT_DEFINE NIFTKQMITKEXT_EXPORT
  INCLUDE_DIRS Events Dialogs LookupTables
  DEPENDS Mitk MitkExt Qmitk QmitkExt niftkMitkExt
  PACKAGE_DEPENDS QT QVTK CTK
  QT_MODULE 
)

TARGET_LINK_LIBRARIES(niftkQmitkExt
  niftkITK
  niftkQt
  )
