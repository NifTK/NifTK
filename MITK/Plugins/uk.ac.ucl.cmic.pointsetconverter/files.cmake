set(SRC_CPP_FILES

)

set(INTERNAL_CPP_FILES
  uk_ac_ucl_cmic_pointsetconverter_Activator.cpp
  PointSetConverterView.cpp
)

set(UI_FILES
  src/internal/PointSetConverterViewControls.ui
)

set(MOC_H_FILES
  src/internal/uk_ac_ucl_cmic_pointsetconverter_Activator.h
  src/internal/PointSetConverterView.h
)

# list of resource files which can be used by the plug-in
# system without loading the plug-ins shared library,
# for example the icon used in the menu and tabs for the
# plug-in views in the workbench
set(CACHED_RESOURCE_FILES
  resources/pointset_interaction.png
  plugin.xml
)

# list of Qt .qrc files which contain additional resources
# specific to this plugin
set(QRC_FILES

)

set(CPP_FILES )

foreach(file ${SRC_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})
