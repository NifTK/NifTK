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
  QmitkSingleWidgetEditor.cpp
)

set(INTERNAL_CPP_FILES
  uk_ac_ucl_cmic_singlewidgeteditor_Activator.cpp
  QmitkSingleWidgetEditorPreferencePage.cpp
)

set(MOC_H_FILES
  src/QmitkSingleWidgetEditor.h
  
  src/internal/uk_ac_ucl_cmic_singlewidgeteditor_Activator.h
  src/internal/QmitkSingleWidgetEditorPreferencePage.h
)

set(UI_FILES

)

set(CACHED_RESOURCE_FILES
  plugin.xml
)

set(QRC_FILES
)

set(CPP_FILES )

foreach(file ${SRC_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})
