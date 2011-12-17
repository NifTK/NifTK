SET(CPP_FILES
  QmitkHelpAboutDialog.cpp
  QmitkMIDASRenderWindow.cpp
  QmitkMIDASSingleViewWidget.cpp
  QmitkMIDASMultiViewWidget.cpp
  QmitkMIDASMultiViewVisibilityManager.cpp
  QmitkMIDASNewSegmentationDialog.cpp
  QmitkThumbnailRenderWindow.cpp
)

SET(MOC_H_FILES
  QmitkMouseEventEater.h 
  QmitkPaintEventEater.h
  QmitkWheelEventEater.h
  QmitkHelpAboutDialog.h
  QmitkMIDASRenderWindow.h
  QmitkMIDASSingleViewWidget.h
  QmitkMIDASMultiViewWidget.h
  QmitkMIDASMultiViewVisibilityManager.h
  QmitkMIDASNewSegmentationDialog.h
  QmitkThumbnailRenderWindow.h
)

SET(UI_FILES
  Resources/UI/QmitkHelpAboutDialog.ui
)

SET(QRC_FILES
  Resources/niftkQmitkExt.qrc
)
