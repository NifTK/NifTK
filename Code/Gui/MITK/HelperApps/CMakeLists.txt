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

set(HELPER_APPS
  # Add apps that DO NOT depend on BUILD_IGI here.
)

if(BUILD_IGI)
  set(HELPER_APPS
    ${HELPER_APPS}
  )
endif()

if(BUILD_PCL)
  set(HELPER_APPS
    ${HELPER_APPS}
  )
endif()

set(HELPER_APP_LIBRARIES "")

list(APPEND HELPER_APP_LIBRARIES
  niftkcommon
  MitkLegacyIO
)

if(BUILD_IGI)
  list(APPEND HELPER_APP_LIBRARIES
    niftkOpenCVUtils
    niftkOpenCV
    niftkIGIGui
    niftkPointReg
    niftkSurfRecon
    niftkICPReg
  )
endif()

if(BUILD_PCL)
  list(APPEND HELPER_APP_LIBRARIES
    niftkPCL
  )
endif()

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
