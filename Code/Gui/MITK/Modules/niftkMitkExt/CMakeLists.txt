MITK_CREATE_MODULE(niftkMitkExt
  EXPORT_DEFINE NIFTKMITKEXT_EXPORT
  INCLUDE_DIRS Algorithms Controllers DataManagement Interactions IO Rendering
  DEPENDS Mitk MitkExt Segmentation PlanarFigure ImageStatistics
  PACKAGE_DEPENDS ITK
)

IF(BUILD_TESTING)
  ADD_SUBDIRECTORY(Testing)
ENDIF(BUILD_TESTING)