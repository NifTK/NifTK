SET(CPP_FILES
  Dialogs/QmitkHelpAboutDialog.cpp
  Dialogs/QmitkMIDASNewSegmentationDialog.cpp
  QmitkMIDASDrawToolGUI.cpp
  QmitkMIDASPaintbrushToolGUI.cpp
  QmitkMIDASSlidersWidget.cpp
  QmitkMIDASOrientationWidget.cpp
  QmitkMIDASStdMultiWidget.cpp
  QmitkMIDASSingleViewWidget.cpp
  QmitkMIDASMultiViewWidget.cpp
  QmitkMIDASMultiViewVisibilityManager.cpp
)

SET(MOC_H_FILES 
  Events/QmitkPaintEventEater.h
  Dialogs/QmitkHelpAboutDialog.h
  Dialogs/QmitkMIDASNewSegmentationDialog.h
  QmitkMIDASDrawToolGUI.h
  QmitkMIDASPaintbrushToolGUI.h
  QmitkMIDASSlidersWidget.h
  QmitkMIDASOrientationWidget.h
  QmitkMIDASStdMultiWidget.h
  QmitkMIDASSingleViewWidget.h
  QmitkMIDASMultiViewWidget.h
  QmitkMIDASMultiViewVisibilityManager.h
)

SET(UI_FILES
  Resources/UI/QmitkHelpAboutDialog.ui
  Resources/UI/QmitkMIDASOrientationWidget.ui
  Resources/UI/QmitkMIDASSlidersWidget.ui
)

SET(QRC_FILES
  Resources/niftkQmitkExt.qrc
)
