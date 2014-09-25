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

###############################################################
# Setup variables describing the library requirements.
###############################################################

set(ITK_NO_IO_FACTORY_REGISTER_MANAGER 0)
include(${MITK_MODULES_PACKAGE_DEPENDS_DIR}/MITK_ITK_Config.cmake)

set(NIFTK_APPLICATION_LINK_LIBRARIES
  niftkcommon
  niftkITK
  ${ITK_LIBRARIES}
  ${Boost_LIBRARIES}
)

find_package(CUDA)
if(CUDA_FOUND)
  set(NIFTK_APPLICATION_WITH_CUDA_LINK_LIBRARIES
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${NIFTK_APPLICATION_LINK_LIBRARIES}
  )
endif()

# Note: vtkRendering seems to force a dependency on Qt. See below.
set(NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES
  niftkVTK
  ${NIFTK_VTK_LIBS_BUT_WITHOUT_QT}
  ${NIFTK_APPLICATION_LINK_LIBRARIES}
)

# Note: vtkRendering seems to force a dependency on Qt. See above.
set(NIFTK_APPLICATION_WITH_VTK_AND_QT_LINK_LIBRARIES
  ${NIFTK_VTK_LIBS_WITH_QT}
  ${QT_LIBRARIES}
  ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES}
)

if (BUILD_MESHING)
  set(NIFTK_MESHING_APPLICATION_LIBRARIES
    niftkMeshingLib
    niftkITK
    niftkVTK
    niftkcommon
    ${ITK_LIBRARIES}
    )
endif (BUILD_MESHING)

set(_all_command_line_apps "")

###############################################################
# These programs use the new SlicerExecutionModel where
# we generate the command line parsing using TCLAP
###############################################################
if(SlicerExecutionModel_FOUND)

  #-------------------------------------------------------------
  # Add new programs to this list.
  #-------------------------------------------------------------
  set(APPS_DEPENDING_ON_SLICER_AND_APPLICATION_LINK_LIBRARIES
    niftkAdd
    niftkAddBorderToImage
    niftkAffine
    niftkAnonymiseDICOMMammograms
    niftkApplyMaskToImage
    niftkBreastDensityCalculationGivenMRISegmentation
    niftkConvertImage
    niftkConvertRawDICOMMammogramsToPresentation
    niftkCreateMaskImage
    niftkCropImage
    niftkDenoise
    niftkDilate
    niftkErode
    niftkExportDICOMTagsToCSVFile
    niftkExtracellularVolumeFractionFromContrastEnhancedCT
    niftkExtract2DSliceFrom3DImage
    niftkExtrudeMaskToVolume
    niftkHistogramEqualization
    niftkHistogramMatchingImageFilter
    niftkImageMomentsRegistration
    niftkInvertImage
    niftkLogInvertImage
    niftkMammogramFatSubtraction
    niftkMammogramMaskSegmentation
    niftkMaskDICOMMammograms
    niftkMammogramPectoralisSegmentation
    niftkMultiply
    niftkN4BiasFieldCorrection
    niftkNegateImage
    niftkOtsuThresholdImage
    niftkRegionalMammographicDensity
    niftkScalarConnectedComponentImageFilter
    niftkSegmentForegroundFromBackground
    niftkSubsampleImage
    niftkSubtract
    niftkUnaryImageOperatorsOnDirectoryTree
    niftkVotingBinaryIterativeHoleFillingImageFilter
  )
  list(APPEND _all_command_line_apps ${APPS_DEPENDING_ON_SLICER_AND_APPLICATION_LINK_LIBRARIES})

  include_directories(${CMAKE_CURRENT_BINARY_DIR})

  foreach(APP ${APPS_DEPENDING_ON_SLICER_AND_APPLICATION_LINK_LIBRARIES})
    NIFTK_CREATE_COMMAND_LINE_APPLICATION(
      NAME ${APP}
      BUILD_SLICER
      INSTALL_SCRIPT
      TARGET_LIBRARIES ${NIFTK_APPLICATION_LINK_LIBRARIES}
    )
  endforeach()

  if(VTK_FOUND)

    #-------------------------------------------------------------
    # Add new programs to this list.
    #-------------------------------------------------------------
    set(APPS_DEPENDING_ON_SLICER_AND_APPLICATION_LINK_LIBRARIES_AND_VTK
      #niftkMakeLapUSProbeBasicModel
      niftkMakeLapUSProbeARUCOModel
      niftkMakeLapUSProbeAprilTagsModel
      niftkMakeLapUSProbeAprilTagsVisualisation
      niftkVTKIterativeClosestPointRegister
      niftkVTKDistanceToSurface
    )
    list(APPEND _all_command_line_apps ${APPS_DEPENDING_ON_SLICER_AND_APPLICATION_LINK_LIBRARIES_AND_VTK})

    include_directories(${CMAKE_CURRENT_BINARY_DIR})

    foreach(APP ${APPS_DEPENDING_ON_SLICER_AND_APPLICATION_LINK_LIBRARIES_AND_VTK})
      NIFTK_CREATE_COMMAND_LINE_APPLICATION(
        NAME ${APP}
        BUILD_SLICER
        INSTALL_SCRIPT
        TARGET_LIBRARIES ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES} vtkIOImage
      )
    endforeach()

  endif(VTK_FOUND)

endif()

###############################################################
# The following applications don't use TCLAP but they do
# produce the SlicerExecutionModel xml code, and hence need
# just the script generating.
###############################################################

set(ADDITIONAL_SLICER_EXECUTION_MODEL_APPS
  #niftkBreastMaskSegmentationFromMRI
  niftkOCTVolumeConstructor
  niftkVesselExtractor
  )
list(APPEND _all_command_line_apps ${ADDITIONAL_SLICER_EXECUTION_MODEL_APPS})

foreach(APP ${ADDITIONAL_SLICER_EXECUTION_MODEL_APPS})
  NIFTK_CREATE_COMMAND_LINE_APPLICATION(
    NAME ${APP}
    INSTALL_SCRIPT
    TARGET_LIBRARIES ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES}
  )
endforeach()


###############################################################
# All other command line apps
###############################################################

##############################################################
# Add new programs to this list.
##############################################################
set(APPS_DEPENDING_ON_NIFTK_APPLICATION_LINK_LIBRARIES
  niftk2DImagesTo3DVolume
  niftkAbsImageFilter
  niftkAnd
  niftkAtlasStatistics
  niftkAverage
  niftkBackProject2Dto3D
  niftkBasicImageFeatures
  niftkBasicImageFeatures3D
  niftkBilateralImageFilter
  niftkBlockMatching
  niftkBreastDicomSeriesReadImageWrite
  niftkBSI
  niftkCombineSegmentations
  niftkComposeITKAffineTransformations
  niftkComputeImageHistogram
  niftkComputeJointHistogram
  niftkComputeMeanTransformation
  niftkConnectedComponents
  #niftkContourExtractor2DImageFilter
  #niftkConvertDofToFsl
  niftkConvertMidasStrToNii
  niftkConvertToMidasStr
  niftkConvertTransformToRIREFormat
  niftkCreateAffineTransform
  niftkCreateMaskFromLabels
  niftkCreateTransformation
  niftkCTEAcosta2009Subsampling
  niftkCTEAssignAtlasValues
  niftkCTEBourgeat2008
  niftkCTEDas2009
  niftkCTEExtractGMWMBoundaryFromLabelImage
  niftkCTEHighRes
  niftkCTEHuttonLayering
  niftkCTEJones2000
  niftkCTEMaskedSmoothing
  niftkCTEPrepareVolumes
  niftkCTEYezzi2003
  niftkCurveFitRegistration
  niftkDecomposeAffineMatrix
  niftkDeformationFieldTargetRegistrationError
  niftkDicomSeriesReadImageWrite
  niftkDilateMaskAndCrop
  niftkDistanceTransform
  niftkDoubleWindowBSI
  niftkExtractCurvatures
  niftkExtractRegion
  niftkExtractScalp
  niftkExtractZeroCrossing
  #niftkFFD
  niftkFillHoles
  niftkForwardAndBackProjectionDifferenceFilter
  niftkForwardProject3Dto2D
  niftkGaussian
  niftkGetMetricValue
  niftkImageInfo
  niftkImageReconstruction
  niftkInject
  niftkInvertAffineTransform
  niftkInvertTransformation
  niftkITKAffineResampleImage
  niftkJacobianStatistics
  niftkKMeansWindowBSI
  niftkKMeansWindowWithLinearRegressionNormalisationBSI
  niftkKNDoubleWindowBSI
  niftkMammogramCharacteristics
  niftkMTPDbc
  niftkMultiplyTransformation
  niftkMultiScaleHessianImageEnhancement2D
  niftkMultiScaleHessianImageEnhancement3D
  niftkOCTVolumeConstructor
  niftkPadImage
  niftkProjectionGeometry
  niftkReorientateImage
  niftkRescale
  niftkResetDirectionField
  niftkResetVoxelDimensionsField
  niftkSegmentationStatistics
  niftkSeriesReadVolumeWrite
  niftkSetBorderPixel
  niftkShiftProb
  niftkShiftScale
  niftkShrinkImage
  niftkSplitVolumeIntoVoxelPlanes
  niftkSTAPLE
  niftkSubtractSliceFromVolume
  niftkSwapIntensity
  niftkTestCompareImage
  niftkTestImage
  niftkThinPlateSplineScatteredDataPointSetToImage
  niftkThreshold
  niftkTransformation
  niftkTransformPoint3Dto2D
  niftkVesselExtractor
  niftkVolToFreeSurfer
  niftkVoxelWiseMaximumIntensities
  niftkKmeansClassifier
)
list(APPEND _all_command_line_apps ${APPS_DEPENDING_ON_NIFTK_APPLICATION_LINK_LIBRARIES})

foreach(APP ${APPS_DEPENDING_ON_NIFTK_APPLICATION_LINK_LIBRARIES})
  NIFTK_CREATE_COMMAND_LINE_APPLICATION(
    NAME ${APP}
    BUILD_CLI
    TARGET_LIBRARIES ${NIFTK_APPLICATION_LINK_LIBRARIES}
  )
endforeach()

##############################################################
# Other command line applications, that require VTK.
##############################################################

if(VTK_FOUND)

  ##############################################################
  # Add new programs to this list.
  ##############################################################
  set(APPS_DEPENDING_ON_NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES
    niftkBreastMaskSegmentationFromMRI
    niftkConvertImageToVTKStructuredGrid
    niftkConvertNiftiVectorImage
    niftkDecimatePolyData
    niftkGradientVectorField
    niftkMapVolumeDataToPolyDataVertices
    niftkMarchingCubes
    niftkSmoothPolyData
    niftkTransformPolyData
  )
  list(APPEND _all_command_line_apps ${APPS_DEPENDING_ON_NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES})

  foreach(APP ${APPS_DEPENDING_ON_NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES})
    NIFTK_CREATE_COMMAND_LINE_APPLICATION(
      NAME ${APP}
      BUILD_CLI
      TARGET_LIBRARIES ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES}
    )
  endforeach()

  if (BUILD_MESHING)
    NIFTK_CREATE_COMMAND_LINE_APPLICATION(
      NAME niftkMeshFromLabels
      BUILD_CLI
      TARGET_LIBRARIES ${NIFTK_MESHING_APPLICATION_LIBRARIES}
    )
  endif (BUILD_MESHING)

endif()

##############################################################
# Other command line applications, that require VTK and QT.
##############################################################

if(VTK_FOUND)
  if(QT_FOUND)

    ##############################################################
    # Add new programs to this list.
    ##############################################################
    set(APPS_DEPENDING_ON_NIFTK_APPLICATION_WITH_VTK_AND_QT_LINK_LIBRARIES
      #niftkCreateBreastMesh
    )
    list(APPEND _all_command_line_apps ${APPS_DEPENDING_ON_NIFTK_APPLICATION_WITH_VTK_AND_QT_LINK_LIBRARIES})

    foreach(APP ${APPS_DEPENDING_ON_NIFTK_APPLICATION_WITH_VTK_AND_QT_LINK_LIBRARIES})
      NIFTK_CREATE_COMMAND_LINE_APPLICATION(
        NAME ${APP}
        BUILD_CLI
        TARGET_LIBRARIES ${NIFTK_APPLICATION_WITH_VTK_AND_QT_LINK_LIBRARIES}
      )
    endforeach()

  endif(QT_FOUND)
endif(VTK_FOUND)

##############################################################
# Other command line applications, that require CUDA.
##############################################################
  
if(CUDA_FOUND AND NIFTK_USE_CUDA)

  ##############################################################
  # Add new programs to this list.
  ##############################################################
  set(APPS_DEPENDING_ON_NIFTK_APPLICATION_WITH_CUDA_LINK_LIBRARIES
    niftkCUDAInfo
  )

  foreach(APP ${APPS_DEPENDING_ON_NIFTK_APPLICATION_WITH_CUDA_LINK_LIBRARIES})
    add_executable(${APP} ${APP}.cxx )
    include_directories(${CUDA_TOOLKIT_INCLUDE})
    target_link_libraries(${APP} ${NIFTK_APPLICATION_WITH_CUDA_LINK_LIBRARIES} )
    install(TARGETS ${APP} RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  endforeach()

endif()

##############################################################
# If no GUI, we still need to make sure we have all the
# necessary libraries for the command line apps to work.
##############################################################
if(NOT BUILD_GUI)
  if(NIFTK_RESOLVE_COMMAND_LINE_DEPENDENCIES)
    NIFTK_FIXUP_COMMAND_LINE_APPLICATIONS(APPS ${_all_command_line_apps})
  endif()
endif()
