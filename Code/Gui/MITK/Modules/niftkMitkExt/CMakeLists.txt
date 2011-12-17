MITK_CREATE_MODULE(niftkMitkExt
  EXPORT_DEFINE NIFTKMITKEXT_EXPORT
  INCLUDE_DIRS Algorithms Controllers DataManagement Interactions IO Rendering
  DEPENDS Mitk MitkExt ipSegmentation PlanarFigure ImageStatistics
  PACKAGE_DEPENDS ITK
)
