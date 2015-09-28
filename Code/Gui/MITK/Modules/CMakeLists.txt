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

# empty first, more added below, depending on conditions.
set(MODULE_DIRS )

# will most likely be used by IGI only, but in itself not limited to that.
if(CUDA_FOUND AND NIFTK_USE_CUDA)
  set(MODULE_DIRS
    ${MODULE_DIRS}
    CUDA
  )
endif()

set(MODULE_DIRS
  ${MODULE_DIRS}
  Core
  CoreIO
  CoreGui
  DnDDisplay
  Thumbnail
)

if(BUILD_MIDAS)
  set(MODULE_DIRS
    ${MODULE_DIRS}
    MIDAS
    MIDASGui
  )
endif()

if(BUILD_PCL)
  set(MODULE_DIRS
    ${MODULE_DIRS}
    PCL
  )
endif()

if(BUILD_IGI)
  set(MODULE_DIRS
    ${MODULE_DIRS}
    OpenCVUtils
    PointReg
    ICPReg
    OpenCV
    IGIServices
    PointRegService
    ICPRegService
    IGI
    IGIGui
    IGIDataSources
    SurfRecon
  )

  if(NVAPI_FOUND AND CUDA_FOUND AND NIFTK_USE_CUDA)
    set(MODULE_DIRS
      ${MODULE_DIRS}
      NVidia
    )
  endif()

  set(MODULE_DIRS
    ${MODULE_DIRS}
    IGIDataSourcesManager
  )
endif()

if(BUILD_VL)
  set(MODULE_DIRS
    ${MODULE_DIRS}
    VL
  )
endif()

foreach(MODULE_DIR ${MODULE_DIRS})
  add_subdirectory(${MODULE_DIR})
endforeach()
