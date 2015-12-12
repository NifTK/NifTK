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
    OpenCVImageConversion
    PointReg
    ICPReg
    OpenCV
    SurfRecon
    IGIServices
    IGI
    IGIGui
    IGITrackers
    IGIDataSources

    # Notice how the IGIDataSourcesManager does NOT know about
    # Each data source. The interface is defined in IGIDataSources
    # but all the implementations are defined after IGIDataSourcesManager.
    IGIDataSourcesManager

    # Micro Services should be last as nothing should depend on them.
    # Furthermore, none of the classes should be exported. All should be internal.
    PointRegService
    ICPRegService
    OIGTLSystemTimeService
    OpenCVVideoDataSourceService
    MITKTrackerDataSourceService
    NiftyLinkDataSourceService
  )

  #if(Ultrasonix_FOUND)
    set(MODULE_DIRS
      ${MODULE_DIRS}
      UltrasonixDataSourceService
    )
  #endif()

 if(QT_QTMULTIMEDIA_INCLUDE_DIR)
    set(MODULE_DIRS
      ${MODULE_DIRS}
      QtAudioDataSourceService
    )
  endif()

  if(NVAPI_FOUND AND CUDA_FOUND AND NIFTK_USE_CUDA)
    set(MODULE_DIRS
      ${MODULE_DIRS}
      NVidia
    )
  endif()

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
