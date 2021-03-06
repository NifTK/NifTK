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
  CoreApp
)

if(BUILD_NiftyMITK)
  set(MODULE_DIRS
    ${MODULE_DIRS}
    # Add more here, if we ever need to
  )
endif()

if(BUILD_NiftyView)
  set(MODULE_DIRS
    ${MODULE_DIRS}
    DnDDisplay
    MIDAS
    MIDASGui
    Thumbnail
  )
endif()

if(BUILD_NiftyMITK)
  set(MODULE_DIRS
    ${MODULE_DIRS}
    Thumbnail
  )
endif()

if(BUILD_NiftyMIDAS)
  set(MODULE_DIRS
    ${MODULE_DIRS}
    DnDDisplay
    MIDAS
    MIDASGui
    Thumbnail
  )
endif()

if(BUILD_PCL)
  set(MODULE_DIRS
    ${MODULE_DIRS}
    PCL
  )
endif()

if(BUILD_NiftyIGI)
  set(MODULE_DIRS
    ${MODULE_DIRS}
    SerialPort
    OpenCVUtils
    OpenCVImageConversion
    QImageConversion
    PointReg
    ICPReg
    OpenCV
    SurfRecon
    IGI
    IGIOverlayEditor
    IGIGui
    IGIServices
    IGIDataSources
    IGITrackers
    NDICAPI
    NDITrackers
    Atracsys
    USRecon
    USReconGui
    USReg
    NiftyCal

    # Notice how the IGIDataSourcesManager does NOT know about
    # Each data source. The interface is defined in IGIDataSources (above)
    # but all the implementations are defined after IGIDataSourcesManager (below).
    IGIDataSourcesManager

    # Micro Services should be last as nothing should depend on them.
    # Furthermore, none of the classes should be exported. All should be internal.
    PointRegService
    ICPRegService
    OIGTLSystemTimeService
    OpenCVVideoDataSourceService
    QtAudioDataSourceService
    UltrasonixDataSourceService
    NDITrackerDataSourceService
    NiftyLinkDataSourceService
    NVidiaSDIDataSourceService
    QtCameraVideoDataSourceService
    BKMedicalDataSourceService
    BlackMagicDataSourceService
    AtracsysDataSourceService
  )

  if(BUILD_Caffe)
    set(MODULE_DIRS
      ${MODULE_DIRS}
      Caffe
      CaffeGui
    )
  endif()

  if(OPENCV_WITH_NONFREE)
    set(MODULE_DIRS
      ${MODULE_DIRS}
      OpenCVGui
    )
  endif()

endif()

if(BUILD_VL)
  set(MODULE_DIRS
    ${MODULE_DIRS}
    VL
  )
endif()

list(REMOVE_DUPLICATES MODULE_DIRS)

foreach(MODULE_DIR ${MODULE_DIRS})
  add_subdirectory(${MODULE_DIR})
endforeach()
