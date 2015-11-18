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

set(HELPER_APP_INCLUDE_DIRECTORIES "")
set(HELPER_APP_LINK_DIRECTORIES "")
set(HELPER_APP_LIBRARIES "")

set(HELPER_APPS_REQUIRED_MODULES
  # Add modules that DO NOT depend on BUILD_IGI here.
)
if(BUILD_IGI)
  list(APPEND HELPER_APPS_REQUIRED_MODULES
    niftkCore
    niftkOpenCVUtils
    niftkOpenCV 
    niftkIGI
    niftkIGIGui
    niftkICPReg
    niftkSurfRecon
    niftkPointReg
  )
endif()

if(BUILD_PCL)
  list(APPEND HELPER_APPS_REQUIRED_MODULES
    niftkPCL
  )
endif()

set(HELPER_APPS
  # Add apps that DO NOT depend on BUILD_IGI here.
)
if(BUILD_IGI)
  set(HELPER_APPS
    ${HELPER_APPS}
  )
  if(BUILD_PCL)
    set(HELPER_APPS
      ${HELPER_APPS}
    )
  endif()
endif()

list(APPEND HELPER_APP_LINK_DIRECTORIES
)

list(APPEND HELPER_APP_LIBRARIES
  niftkcommon
  niftkOpenCVUtils
  niftkOpenCV
  niftkIGIGui
  niftkPointReg
  niftkSurfRecon
  niftkICPReg
  MitkLegacyIO
)

if(BUILD_PCL)
  list(APPEND HELPER_APP_LIBRARIES
    niftkPCL
  )
endif()

list(REMOVE_DUPLICATES HELPER_APP_INCLUDE_DIRECTORIES)
list(REMOVE_DUPLICATES HELPER_APP_LINK_DIRECTORIES)
list(REMOVE_DUPLICATES HELPER_APP_LIBRARIES)

include_directories(${HELPER_APP_INCLUDE_DIRECTORIES})
link_directories(${HELPER_APP_LINK_DIRECTORIES})

# Then foreach app, we just build against libraries (modules).
foreach(APP ${HELPER_APPS})
  NIFTK_CREATE_COMMAND_LINE_APPLICATION(
    NAME ${APP}
    BUILD_CLI
    TARGET_LIBRARIES ${HELPER_APP_LIBRARIES}
  )
endforeach()

# Repeat the same process for slicer apps.
if(SlicerExecutionModel_FOUND)
  set(SLICER_HELPER_APPS
    # Add Slicer apps that DO NOT depend on BUILD_IGI here.
    niftkPointSetStatistics
  )

  if(BUILD_IGI)
    set(SLICER_HELPER_APPS
      ${SLICER_HELPER_APPS}
      niftkCameraCalibration
      niftkCorrectVideoDistortion
      niftkProject3DPointsToStereoPair
      niftkReviewVideoData
      niftkHandeyeCalibration
      niftkHandeyeCalibrationFromDirectory
      niftkHandeyeCalibrationUsingRegistration
      niftkGetMatchedTrackingMatrix
      niftkLaparoscopeTrackingAccuracy
      niftkTwoTrackerAnalysis
      niftkProjectTrackedPointsOnStereoVideo
      niftkPickPointsOnStereoVideo
      niftkMakeMaskImagesFromStereoVideo
      niftkSplitVideo
      niftkFindAndTriangulateCrossHair
      niftkVideoToImages
      niftkVideoToSurfaces
      niftkUltrasoundPinCalibrationClicker
      niftkUltrasoundPinCalibration
      niftkVideoHandEyeByCrossCalibration
      #niftkUltrasoundPointerCalibration
      #niftkUltrasoundPinCalibrationEvaluation
      niftkUltrasoundTransformAndImageMerger
      niftkTriangulate2DPointPairsTo3D
      niftkPivotCalibration
      niftkIGIMakeGeometry
      niftkMakeGridOf2DImages
      niftkIterativeClosestPointRegister
      niftkPointSetRegister
      niftkPointSetTransform
      # this one does not depend on pcl!
      niftkMergePointClouds
      niftkAverageTrackingMatrices
      niftkAverageStationaryChessboards
      niftkEvaluateIntrinsicParametersOnNumberOfFrames
      niftkMakeChessBoardPointSet
      niftkBifurcationToPointSet
      niftkMakeLapUSProbeAprilTagsModel
      niftkMakeLapUSProbeSimulationData
      niftkImageFeatureMatching
      niftkVideoTestClient
    )

    if(BUILD_PCL)
      set(SLICER_HELPER_APPS
        ${SLICER_HELPER_APPS}
        niftkFitPlaneToPointCloud
      )
    endif()

    if(OPENCV_WITH_NONFREE)
      set(SLICER_HELPER_APPS
        ${SLICER_HELPER_APPS}
        niftkSURF
      )
    endif()

  endif()

  set(SLICER_HELPER_APP_LIBRARIES
    ${HELPER_APP_LIBRARIES}
    ${Boost_LIBRARIES}
  )

  if(BUILD_IGI)
    list(APPEND SLICER_HELPER_APP_LIBRARIES
      niftkVTK
    )
  endif()

  foreach(APP ${SLICER_HELPER_APPS})
    NIFTK_CREATE_COMMAND_LINE_APPLICATION(
      NAME ${APP}
      BUILD_SLICER
      INSTALL_SCRIPT
      TARGET_LIBRARIES ${SLICER_HELPER_APP_LIBRARIES}
    )
  endforeach()

  if(BUILD_IGI)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      # Suppress warning because of a bug in the Boost gil library.
      set_target_properties(niftkImageFeatureMatching PROPERTIES COMPILE_FLAGS "-Wno-c++11-narrowing")
    endif()
  endif()

endif()
