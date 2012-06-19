SET(SRC_CPP_FILES  
)

SET(INTERNAL_CPP_FILES
  SurfaceExtractorView.cpp
  SurfaceExtractorPreferencePage.cpp
  SurfaceExtractorPluginActivator.cpp
)

SET(UI_FILES
  src/internal/SurfaceExtractorViewControls.ui
)

SET(MOC_H_FILES
  src/internal/SurfaceExtractorView.h
  src/internal/SurfaceExtractorPreferencePage.h
  src/internal/SurfaceExtractorPluginActivator.h
)

SET(CACHED_RESOURCE_FILES
  resources/SurfaceExtractor.png
  plugin.xml
# list of resource files which can be used by the plug-in
# system without loading the plug-ins shared library,
# for example the icon used in the menu and tabs for the
# plug-in views in the workbench
)

SET(QRC_FILES
  resources/SurfaceExtractor.qrc
)

SET(CPP_FILES 
)

foreach(file ${SRC_CPP_FILES})
  SET(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  SET(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})