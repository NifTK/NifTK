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

set(MODULE_DIRS
  niftkCore
  niftkCoreGui
)

if(BUILD_MIDAS)

  set(MODULE_DIRS
    ${MODULE_DIRS}
    niftkDnDDisplay
    niftkMIDAS
    niftkMIDASGui
  )

endif()

if(BUILD_IGI)

  set(MODULE_DIRS
    ${MODULE_DIRS}
    niftkOpenCV
    niftkIGI
    niftkIGIGui
  )

  if(NVAPI_FOUND)
    set(MODULE_DIRS
      ${MODULE_DIRS}
      niftkNVidia
      niftkNVidiaGui
    )
  endif()

  set(MODULE_DIRS
    ${MODULE_DIRS}
    niftkIGIGuiManager
  )

endif()

foreach(MODULE_DIR ${MODULE_DIRS})
  add_subdirectory(${MODULE_DIR})
endforeach()
