#
# MITK modules and plugins that are allowed to build when building MITK for NiftyIGI.
#
# Listing a module or plugin here does not mean that it will actually be built, that you can
# control through CMake flags when configuring MITK. However, if a required module or plugin
# is not listed here, it will not be built and it might prevent other modules or plugins from
# being built that you would need. (If the required module depends on a module that is not
# listed, eventually transitively.) The dependency lists are not complete, only the first
# dependency is marked in the comments.

set(enabled_modules

# Modules with linking dependency from our code:

  Core                    # needed by niftkCore
  SceneSerializationBase  # needed by niftkCoreIO
  LegacyGL                # needed by PlanarFigure and AlgorithmsExt
  PlanarFigure            # needed by QtWidgets
  Overlays                # needed by QtWidgets
  QtWidgets               # needed by niftkCoreGui
  DataTypesExt            # needed by AlgorithmsExt
  AlgorithmsExt           # needed by ImageExtraction
  ImageExtraction         # needed by ImageStatistics
  ImageStatistics         # needed by QtWidgetsExt
  QtWidgetsExt            # needed by niftkCoreGui
  SceneSerialization      # needed by org.mitk.gui.qt.ext
  AppUtil                 # needed by org.mitk.gui.qt.ext
  LegacyIO                # needed by uk.ac.ucl.cmic.xnat
  LegacyAdaptors          # needed by Segmentation
  SurfaceInterpolation    # needed by Segmentation
  GraphAlgorithms         # needed by Segmentation
  ContourModel            # needed by Segmentation
  Multilabel              # needed by Segmentation
  Segmentation            # needed by uk.ac.ucl.cmic.surfaceextractor

  IGTBase                 # needed by IGT
  OpenIGTLink             # needed by IGT
  IGT                     # needed by CameraCalibration
  OpenCVVideoSupport      # needed by niftkOpenCV
  CameraCalibration       # needed by niftkOpenCV
  Persistence             # needed by IGTUI
  IGTUI                   # needed by niftkIGIGui

# Auto-load modules. No linking dependency, but they provide IO classes and mappers that we need.

  IOExt
#  IpPicSupportIO
#  VtkShaders

)

set(enabled_plugins

# Plugins with compile-time or run-time dependency from our code:

  org.blueberry.core.runtime      # needed by org.blueberry.core.commands
  org.blueberry.core.commands     # needed by org.blueberry.ui.qt
  org.blueberry.core.expressions  # needed by org.blueberry.ui.qt
  org.blueberry.ui.qt             # needed by NiftyIGI
  org.mitk.core.ext               # needed by org.mitk.gui.qt.ext
  org.mitk.core.services          # needed by org.mitk.gui.common
  org.mitk.gui.common             # needed by org.mitk.gui.qt.application
  org.mitk.gui.qt.application     # needed by org.mitk.gui.qt.datamanager and org.mitk.gui.qt.ext
  org.mitk.gui.qt.common          # needed by org.mitk.gui.qt.datamanager
  org.mitk.gui.qt.ext             # needed by NiftyIGI
  org.mitk.gui.qt.datamanager     # needed by NiftyIGI

# Plugins that we do not physically depend one but that we want to use:

  org.blueberry.ui.qt.help
  org.blueberry.ui.qt.log
  org.mitk.gui.qt.properties

)
