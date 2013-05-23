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
  QmitkBaseView.cxx
)

set(INTERNAL_CPP_FILES
  CommonActivator.cxx
  VisibilityChangedCommand.cxx
  VisibilityChangeObserver.cxx
)

set(UI_FILES
)

set(MOC_H_FILES
  src/internal/CommonActivator.h
  src/QmitkBaseView.h
)

set(CACHED_RESOURCE_FILES
  plugin.xml
)

# todo: add some qt style sheet resources
set(QRC_FILES
  resources/CommonResources.qrc
)

set(CPP_FILES )

foreach(file ${SRC_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})
