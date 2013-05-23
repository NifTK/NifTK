set(SRC_CPP_FILES    
)

set(INTERNAL_CPP_FILES
  QmitkImageLookupTablesPreferencePage.cxx
  ImageLookupTablesViewActivator.cxx
  ImageLookupTablesView.cxx
)

set(UI_FILES
  src/internal/ImageLookupTablesViewControls.ui
)

set(MOC_H_FILES
  src/internal/QmitkImageLookupTablesPreferencePage.h
  src/internal/ImageLookupTablesViewActivator.h
  src/internal/ImageLookupTablesView.h
)

set(CACHED_RESOURCE_FILES
  resources/ImageLookupTables.png
  plugin.xml
# list of resource files which can be used by the plug-in
# system without loading the plug-ins shared library,
# for example the icon used in the menu and tabs for the
# plug-in views in the workbench
)

set(QRC_FILES
  resources/ImageLookupTables.qrc
)

set(CPP_FILES 
)

foreach(file ${SRC_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})
