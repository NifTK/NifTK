#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

set(SRC_CPP_FILES
  QmitkCommonAppsApplicationPlugin.cxx
  QmitkCommonAppsApplicationPreferencePage.cxx
  QmitkBaseApplication.cxx
  QmitkBaseAppWorkbenchAdvisor.cxx
  QmitkBaseWorkbenchWindowAdvisor.cxx
  QmitkNiftyViewApplicationPreferencePage.cxx
  QmitkCommonAppsMinimalPerspective.cxx
  QmitkCommonAppsIGIPerspective.cxx
  QmitkMIDASSegmentationPerspective.cxx

  QmitkExtActionBarAdvisor.cpp
  QmitkExtWorkbenchWindowAdvisor.cpp
  QmitkExtFileSaveProjectAction.cpp
  QmitkOpenDicomEditorAction.cpp
  QmitkOpenXnatEditorAction.cpp
)

set(INTERNAL_CPP_FILES
  QmitkAppInstancesPreferencePage.cpp
  QmitkModuleView.cpp
)

set(UI_FILES
  src/internal/QmitkAppInstancesPreferencePage.ui
)

set(MOC_H_FILES
  src/QmitkCommonAppsApplicationPlugin.h
  src/QmitkCommonAppsApplicationPreferencePage.h
  src/QmitkBaseApplication.h
  src/QmitkBaseWorkbenchWindowAdvisor.h
  src/QmitkNiftyViewApplicationPreferencePage.h
  src/QmitkCommonAppsMinimalPerspective.h
  src/QmitkCommonAppsIGIPerspective.h
  src/QmitkMIDASSegmentationPerspective.h

  src/QmitkExtFileSaveProjectAction.h
  src/QmitkExtWorkbenchWindowAdvisor.h

  src/internal/QmitkAppInstancesPreferencePage.h
  src/internal/QmitkExtWorkbenchWindowAdvisorHack.h
  src/internal/QmitkModuleView.h
  src/QmitkOpenDicomEditorAction.h
  src/QmitkOpenXnatEditorAction.h
)

set(CACHED_RESOURCE_FILES
# list of resource files which can be used by the plug-in
# system without loading the plug-ins shared library,
# for example the icon used in the menu and tabs for the
# plug-in views in the workbench
  plugin.xml
  resources/icon_cmic.xpm
  resources/icon_ion.xpm
  resources/icon_ucl.xpm

  resources/ModuleView.png
)

set(QRC_FILES
# uncomment the following line if you want to use Qt resources
  resources/CommonAppsResources.qrc

# Note:
# Some features of the org.mitk.gui.qt.ext plugin has been merged into the
# current plugin. To minimise differences between the duplicated code, we
# leave the resource file with its original name and contents.
  resources/org_mitk_gui_qt_ext.qrc
  resources/org_mitk_icons.qrc
)

set(CPP_FILES )

foreach(file ${INTERNAL_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})

foreach(file ${SRC_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})
