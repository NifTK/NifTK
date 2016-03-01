set(SRC_CPP_FILES

)

set(INTERNAL_CPP_FILES
  mitkPointSetCropperPluginActivator.cpp
  QmitkPointSetCropper.cpp
  mitkPointSetCropperEventInterface.cpp
)

set(UI_FILES
  src/internal/QmitkPointSetCropperControls.ui
)

set(MOC_H_FILES
  src/internal/mitkPointSetCropperPluginActivator.h
  src/internal/QmitkPointSetCropper.h
)

set(CACHED_RESOURCE_FILES
  resources/icon.xpm
  plugin.xml
)

set(QRC_FILES
  resources/pointsetcropper.qrc
)

set(CPP_FILES)

foreach(file ${SRC_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})
