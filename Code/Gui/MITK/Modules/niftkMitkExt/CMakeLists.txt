MITK_CREATE_MODULE(niftkMitkExt
  EXPORT_DEFINE NIFTKMITKEXT_EXPORT
  INCLUDE_DIRS Algorithms Common Controllers DataManagement DataNodeProperties Interactions IO Rendering
  DEPENDS Mitk MitkExt Segmentation PlanarFigure ImageStatistics
  PACKAGE_DEPENDS ITK
)

TARGET_LINK_LIBRARIES(niftkMitkExt
  niftkcommon
  niftkITK
  )
  
IF(BUILD_TESTING)
  ADD_SUBDIRECTORY(Testing)
ENDIF(BUILD_TESTING)