set(SRC_CPP_FILES
  
)

set(INTERNAL_CPP_FILES
  NiftyRegPluginActivator.cpp
  QmitkNiftyRegView.cpp
  mitkImageToNifti.cpp
  niftiImageToMitk.cpp
  RegistrationExecution.cpp
)

set(UI_FILES
  src/internal/QmitkNiftyRegViewControls.ui
)

set(MOC_H_FILES
  src/internal/NiftyRegPluginActivator.h
  src/internal/QmitkNiftyRegView.h
  src/internal/mitkImageToNifti.h
  src/internal/niftiImageToMitk.h
  src/internal/RegistrationExecution.h
)

# list of resource files which can be used by the plug-in
# system without loading the plug-ins shared library,
# for example the icon used in the menu and tabs for the
# plug-in views in the workbench
set(CACHED_RESOURCE_FILES
  plugin.xml
  resources/NiftyRegLogo.png
)

# list of Qt .qrc files which contain additional resources
# specific to this plugin
set(QRC_FILES
# uncomment the following line if you want to use Qt resources
  resources/NiftyReg.qrc
)

set(CPP_FILES )

foreach(file ${SRC_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})

