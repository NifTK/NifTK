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

#################################################################################
# Command line applications, integrated into GUI via the CLI module.
#################################################################################

if(BUILD_COMMAND_LINE_PROGRAMS)

  if(SlicerExecutionModel_FOUND)

    add_subdirectory(PointSetStatistics)
    add_subdirectory(CreatePolyDataFromImage)

    if(BUILD_NiftyIGI)
      add_subdirectory(AverageStationaryChessboards)
      add_subdirectory(AverageTrackingMatrices)
      add_subdirectory(BifurcationToPointSet)
      add_subdirectory(CameraCalibration)
      add_subdirectory(CorrectVideoDistortion)
      add_subdirectory(EvaluateIntrinsicParams)
      add_subdirectory(FindAndTriangulateCrossHair)
      add_subdirectory(GetMatchedTrackingMatrix)
      add_subdirectory(GetTrackerStats)
      add_subdirectory(HandeyeCalibration)
      add_subdirectory(HandeyeCalibrationFromDirectory)
      add_subdirectory(HandeyeCalibrationUsingRegistration)
      add_subdirectory(IGIMakeGeometry)
      add_subdirectory(ImageFeatureMatching)
      add_subdirectory(IterativeClosestPointRegister)
      add_subdirectory(LaparoscopeTrackingAccuracy)
      add_subdirectory(MakeChessBoardPointSet)
      add_subdirectory(MakeGridOf2DImages)
      add_subdirectory(MakeLapUSProbeAprilTagsModel)
      add_subdirectory(MakeLapUSProbeSimulationData)
      add_subdirectory(MakeMaskImagesFromStereoVideo)
      # this one does not depend on PCL!
      add_subdirectory(MergePointClouds)
      add_subdirectory(PickPointsOnStereoVideo)
      add_subdirectory(PivotCalibration)
      add_subdirectory(PointSetRegister)
      add_subdirectory(PointSetTransform)
      add_subdirectory(Project3DPointsToStereoPair)
      add_subdirectory(ProjectTrackedPointsOnStereoVideo)
      add_subdirectory(ReviewVideoData)
      add_subdirectory(SegmentLiverPhantom)
      add_subdirectory(CalculateReProjectionErrors)
      add_subdirectory(SplitVideo)
      add_subdirectory(TimingCalibration)
      add_subdirectory(Triangulate2DPointPairsTo3D)
      add_subdirectory(TwoTrackerAnalysis)
      add_subdirectory(UltrasoundPinCalibration)
      add_subdirectory(CalibrationPointClicker)
      add_subdirectory(UltrasoundBallCalibration)
      add_subdirectory(UltrasoundReconstruction)
      #add_subdirectory(UltrasoundPinCalibrationEvaluation)
      #add_subdirectory(UltrasoundPointerCalibration)
      add_subdirectory(UltrasoundTransformAndImageMerger)
      add_subdirectory(VideoHandEyeByCrossCalibration)
      add_subdirectory(VideoTestClient)
      add_subdirectory(VideoToImages)
      add_subdirectory(VideoToSurfaces)

      if(BUILD_PCL)
        add_subdirectory(FitPlaneToPointCloud)
      endif()

      if(OPENCV_WITH_NONFREE)
        add_subdirectory(SURF)
      endif()

      if(BUILD_CAFFE)
        add_subdirectory(CaffeSeg)
      endif()

    endif()

  endif()

endif()

#################################################################################
# GUI applications based on the BlueBerry framework.
#
# This section must go after the command line applications, or else when you run
# the CMake fixup_bundle macro (most important for Mac), the command line apps 
# are not yet present, and the command line apps end up with unresolved external 
# dependencies. Worse still, the packaging will appear to succeed.
#################################################################################

# Note: The name of the App in Apps.cmake, must match the directory name.

if(MITK_USE_BLUEBERRY)

  foreach(NIFTK_APP ${NIFTK_APPS})

    # extract option_name
    string(REPLACE "^^" "\\;" target_info ${NIFTK_APP})
    set(target_info_list ${target_info})
    list(GET target_info_list 1 option_name)
    list(GET target_info_list 0 app_name)

    if(${option_name})
      add_subdirectory(${app_name})
    endif()

  endforeach()

endif()

