SET(SRC_CPP_FILES    
)

SET(INTERNAL_CPP_FILES
  LookupTableContainer.cpp
  LookupTableSaxHandler.cpp
  LookupTableManager.cpp
  NamedLookupTableProperty.cpp
  QmitkImageLookupTablesPreferencePage.cpp
  ImageLookupTablesView.cpp
  ImageLookupTablesViewActivator.cpp
)

SET(UI_FILES
  src/internal/ImageLookupTablesViewControls.ui
)

SET(MOC_H_FILES
  src/internal/QmitkImageLookupTablesPreferencePage.h
  src/internal/ImageLookupTablesView.h
  src/internal/ImageLookupTablesViewActivator.h
)

SET(CACHED_RESOURCE_FILES
  resources/ImageLookupTables.png
  plugin.xml
# list of resource files which can be used by the plug-in
# system without loading the plug-ins shared library,
# for example the icon used in the menu and tabs for the
# plug-in views in the workbench
  resources/blue.lut
  resources/cyan.lut
  resources/green.lut
  resources/grey.lut
  resources/hot.lut
  resources/hsv.lut
  resources/imagej_fire.lut
  resources/jet.lut
  resources/magenta.lut
  resources/matlab_autumn.lut
  resources/matlab_bipolar_256_0.1.lut
  resources/matlab_bipolar_256_0.9.lut
  resources/matlab_cool.lut
  resources/matlab_hot.lut
  resources/matlab_spring.lut
  resources/matlab_summer.lut
  resources/matlab_winter.lut
  resources/midas_bands.lut
  resources/midas_hot_iron.lut
  resources/midas_overlay.lut
  resources/midas_pet_map.lut
  resources/midas_spectrum.lut
  resources/nih.lut
  resources/red.lut
  resources/sea.lut
  resources/yellow.lut
)

SET(QRC_FILES
  resources/ImageLookupTables.qrc
)

SET(CPP_FILES 
)

foreach(file ${SRC_CPP_FILES})
  SET(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  SET(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})
