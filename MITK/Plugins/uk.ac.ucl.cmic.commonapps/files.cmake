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
  niftkCommonAppsApplicationPlugin.cxx
  niftkCommonAppsApplicationPreferencePage.cxx
  niftkBaseApplication.cxx
  niftkBaseAppWorkbenchAdvisor.cxx
  niftkBaseWorkbenchWindowAdvisor.cxx
  niftkNiftyViewApplicationPreferencePage.cxx
  niftkCommonAppsMinimalPerspective.cxx
  niftkCommonAppsIGIPerspective.cxx
)

set(INTERNAL_CPP_FILES
)

set(MOC_H_FILES
  src/niftkCommonAppsApplicationPlugin.h
  src/niftkCommonAppsApplicationPreferencePage.h
  src/niftkBaseApplication.h
  src/niftkBaseWorkbenchWindowAdvisor.h
  src/niftkNiftyViewApplicationPreferencePage.h
  src/niftkCommonAppsMinimalPerspective.h
  src/niftkCommonAppsIGIPerspective.h
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
)

set(QRC_FILES
# uncomment the following line if you want to use Qt resources
  resources/CommonAppsResources.qrc
)

set(CPP_FILES )

foreach(file ${INTERNAL_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})

foreach(file ${SRC_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})
