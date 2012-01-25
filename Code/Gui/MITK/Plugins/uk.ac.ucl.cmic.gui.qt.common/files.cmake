SET(SRC_CPP_FILES
  mitkMIDASDataStorageEditorInput.cpp
  QmitkFunctionalityWithoutStdMultiWidget.cpp
  QmitkCMICBaseFunctionality.cpp
  QmitkMIDASBaseFunctionality.cpp
  QmitkMIDASBaseSegmentationFunctionality.cpp
  QmitkMIDASImageAndSegmentationSelectorWidget.cpp
  QmitkMIDASMultiViewEditor.cpp
  QmitkMIDASMultiViewEditorPreferencePage.cpp
)

SET(INTERNAL_CPP_FILES
  CommonActivator.cpp
  QmitkFunctionalityUtil.cpp
)

SET(UI_FILES
  src/QmitkMIDASImageAndSegmentationSelector.ui
)

SET(MOC_H_FILES
  src/internal/CommonActivator.h
  src/QmitkCMICBaseFunctionality.h
  src/QmitkMIDASBaseFunctionality.h
  src/QmitkMIDASBaseSegmentationFunctionality.h
  src/QmitkMIDASImageAndSegmentationSelectorWidget.h
  src/QmitkMIDASMultiViewEditor.h
  src/QmitkMIDASMultiViewEditorPreferencePage.h
)

SET(CACHED_RESOURCE_FILES
  plugin.xml
)

# todo: add some qt style sheet resources
SET(QRC_FILES
  resources/CommonResources.qrc
)

SET(CPP_FILES )

foreach(file ${SRC_CPP_FILES})
  SET(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  SET(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})
