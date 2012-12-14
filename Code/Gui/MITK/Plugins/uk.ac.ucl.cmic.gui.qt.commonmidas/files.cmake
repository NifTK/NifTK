SET(SRC_CPP_FILES
  QmitkMIDASBaseSegmentationFunctionality.cpp
  QmitkMIDASSegmentationViewWidget.cpp
)

SET(INTERNAL_CPP_FILES
  MIDASActivator.cpp
)

SET(UI_FILES
  src/QmitkMIDASSegmentationViewWidget.ui
)

SET(MOC_H_FILES
  src/internal/MIDASActivator.h
  src/QmitkMIDASBaseSegmentationFunctionality.h
  src/QmitkMIDASSegmentationViewWidget.h
)

SET(CACHED_RESOURCE_FILES
  plugin.xml
)

# todo: add some qt style sheet resources
SET(QRC_FILES
  resources/CommonMIDASResources.qrc
)

SET(CPP_FILES )

foreach(file ${SRC_CPP_FILES})
  SET(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  SET(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})
