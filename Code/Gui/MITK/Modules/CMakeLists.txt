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

SET(MODULE_DIRS
  niftkMitkExt
  niftkQmitkExt
  niftkCore
  niftkCoreGui
  niftkMIDAS
  niftkMIDASGui
  XnatRest
  XnatRestWidgets
)

IF(BUILD_IGI)

  SET(MODULE_DIRS
    ${MODULE_DIRS}
    niftkOpenCV
    niftkIGI
    niftkIGIGui
  )

  IF(NVAPI_FOUND)  
    SET(MODULE_DIRS
      ${MODULE_DIRS}
      niftkNVidia
      niftkNVidiaGui
    )
  ENDIF()

  SET(MODULE_DIRS
    ${MODULE_DIRS}
    niftkIGIGuiManager    
  )
  
ENDIF()

FOREACH(MODULE_DIR ${MODULE_DIRS})
  ADD_SUBDIRECTORY(${MODULE_DIR})
ENDFOREACH()
