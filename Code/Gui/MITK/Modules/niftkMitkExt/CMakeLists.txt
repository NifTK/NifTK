MITK_CREATE_MODULE(niftkMitkExt
  EXPORT_DEFINE NIFTKMITKEXT_EXPORT
  INCLUDE_DIRS Algorithms Common Controllers DataManagement Interactions IO Rendering
  DEPENDS Mitk MitkExt Segmentation PlanarFigure ImageStatistics
  PACKAGE_DEPENDS ITK
)

TARGET_LINK_LIBRARIES(niftkMitkExt
  niftkcommon
  )
  
IF(BUILD_TESTING)
  ADD_SUBDIRECTORY(Testing)
ENDIF(BUILD_TESTING)