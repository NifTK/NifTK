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
  Thumbnail
)

if(BUILD_MIDAS)
  set(MODULE_DIRS
    ${MODULE_DIRS}
    DnDDisplay
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
    SerialPort
    NDICAPI 
    OpenCVUtils
    OpenCVImageConversion
    QImageConversion
    PointReg
    ICPReg
    OpenCV
    SurfRecon
    NiftyCal
    NDITrackers
    IGI
    IGIOverlayEditor
    IGIGui
    IGIServices
    IGIDataSources

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
  )

  if(BUILD_CAFFE)
    set(MODULE_DIRS
      ${MODULE_DIRS}
      Caffe
      CaffeGui
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
