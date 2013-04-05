set(SRC_CPP_FILES
)

set(INTERNAL_CPP_FILES
  it_unito_cim_intensityprofile_Activator.cxx
  IntensityProfileView.cxx
  PropagateSegmentationAlongTimeAction.cxx
  ItkProcessObserver.cxx
)

set(UI_FILES
  src/internal/IntensityProfileView.ui
)

set(MOC_H_FILES
  src/internal/it_unito_cim_intensityprofile_Activator.h
  src/internal/IntensityProfileView.h
  src/internal/PropagateSegmentationAlongTimeAction.h
)

# list of resource files which can be used by the plug-in
# system without loading the plug-ins shared library,
# for example the icon used in the menu and tabs for the
# plug-in views in the workbench
set(CACHED_RESOURCE_FILES
  plugin.xml
  resources/intensity_profile_icon.jpg
)

# list of Qt .qrc files which contain additional resources
# specific to this plugin
set(QRC_FILES
  resources/IntensityProfileResources.qrc
)

set(CPP_FILES)

foreach(file ${SRC_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})

