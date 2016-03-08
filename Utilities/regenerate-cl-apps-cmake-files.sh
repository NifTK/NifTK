#!/bin/bash

#
# The script was used to move the CL applications from one directory with a long
# monolite CMake file to their individual directories. This allows finer control
# of their dependencies and it looks cleaner.
#

for app in niftkAdd \
    niftkAddBorderToImage\
    niftkAffine\
    niftkAnonymiseDICOMMammograms\
    niftkAnonymiseDICOMImages\
    niftkApplyMaskToImage\
    niftkBreastDensityCalculationGivenMRISegmentation\
    niftkConvertImage\
    niftkConvertImageToDICOM\
    niftkConvertRawDICOMMammogramsToPresentation\
    niftkCreateMaskImage\
    niftkCropImage\
    niftkDenoise\
    niftkDilate\
    niftkElasticBodySplineWarp\
    niftkErode\
    niftkExportDICOMTagsToCSVFile\
    niftkExtracellularVolumeFractionFromContrastEnhancedCT\
    niftkExtract2DSliceFrom3DImage\
    niftkExtrudeMaskToVolume\
    niftkHistogramEqualization\
    niftkHistogramMatchingImageFilter\
    niftkImageMomentsRegistration\
    niftkInvertImage\
    niftkLogInvertImage\
    niftkMammogramFatSubtraction\
    niftkMammogramMaskSegmentation\
    niftkMaskDICOMMammograms\
    niftkMammogramPectoralisSegmentation\
    niftkMultiply\
    niftkN4BiasFieldCorrection\
    niftkNegateImage\
    niftkOtsuThresholdImage\
    niftkPrintDICOMSeries\
    niftkRescaleImageUsingHistogramPercentiles\
    niftkScalarConnectedComponentImageFilter\
    niftkSegmentForegroundFromBackground\
    niftkSubsampleImage\
    niftkSubtract\
    niftkThinPlateSplineScatteredDataPointSetToImage\
    niftkThinPlateSplineWarp\
    niftkUnaryImageOperatorsOnDirectoryTree\
    niftkVesselExtractor\
    niftkVotingBinaryIterativeHoleFillingImageFilter
do

  appname=${app:5}

echo "#/*============================================================================
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

if(SlicerExecutionModel_FOUND)

  NIFTK_CREATE_COMMAND_LINE_APPLICATION(
    NAME $app
    BUILD_SLICER
    INSTALL_SCRIPT
    TARGET_LIBRARIES
      niftkcommon
      niftkITK
      niftkITKIO
      \${ITK_LIBRARIES}
      \${Boost_LIBRARIES}
  )

endif()
" > $appname/CMakeLists.txt

done




for app in \
      niftkMakeLapUSProbeBasicModel\
      niftkMakeLapUSProbeARUCOModel\
      niftkMakeLapUSProbeAprilTagsVisualisation\
      niftkVTKIterativeClosestPointRegister\
      niftkVTKDistanceToSurface
do

  appname=${app:5}

echo "#/*============================================================================
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

if(SlicerExecutionModel_FOUND AND VTK_FOUND)

  NIFTK_CREATE_COMMAND_LINE_APPLICATION(
    NAME $app
    BUILD_SLICER
    INSTALL_SCRIPT
    TARGET_LIBRARIES
      niftkVTK
      \${NIFTK_VTK_LIBS_BUT_WITHOUT_QT}
      niftkcommon
      niftkITK
      niftkITKIO
      \${ITK_LIBRARIES}
      \${Boost_LIBRARIES}
  )

endif()
" > $appname/CMakeLists.txt

done



for app in \
        niftkBreastDCEandADC\
        niftkBreastDensityFromMRIs\
        niftkBreastDensityFromMRIsGivenMaskAndImage\
        niftkMammographicTumourDistribution\
        niftkRegionalMammographicDensity
do

  appname=${app:5}

echo "#/*============================================================================
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

if(SlicerExecutionModel_FOUND AND VTK_FOUND AND MITK_USE_Qt4)

  NIFTK_CREATE_COMMAND_LINE_APPLICATION(
    NAME $app
    BUILD_SLICER
    INSTALL_SCRIPT
    TARGET_LIBRARIES
      \${NIFTK_VTK_LIBS_WITH_QT}
      \${QT_LIBRARIES}
      niftkVTK
      \${NIFTK_VTK_LIBS_BUT_WITHOUT_QT}
      niftkcommon
      niftkITK
      niftkITKIO
      \${ITK_LIBRARIES}
      \${Boost_LIBRARIES}
  )

endif()
" > $appname/CMakeLists.txt

done



for app in \
  niftk2DImagesTo3DVolume\
  niftkAbsImageFilter\
  niftkAnd\
  niftkAtlasStatistics\
  niftkAverage\
  niftkBackProject2Dto3D\
  niftkBasicImageFeatures\
  niftkBasicImageFeatures3D\
  niftkBilateralImageFilter\
  niftkBinaryShapeBasedSuperSamplingFilter\
  niftkBlockMatching\
  niftkBreastDicomSeriesReadImageWrite\
  niftkBSI\
  niftkCombineSegmentations\
  niftkComposeITKAffineTransformations\
  niftkComputeImageHistogram\
  niftkComputeJointHistogram\
  niftkComputeMeanTransformation\
  niftkConnectedComponents\
  niftkContourExtractor2DImageFilter\
  niftkConvertMidasStrToNii\
  niftkConvertToMidasStr\
  niftkConvertTransformToRIREFormat\
  niftkCreateAffineTransform\
  niftkCreateAffineTransform2D\
  niftkCreateMaskFromLabels\
  niftkCreateTransformation\
  niftkCTEAcosta2009Subsampling\
  niftkCTEAssignAtlasValues\
  niftkCTEBourgeat2008\
  niftkCTEDas2009\
  niftkCTEExtractGMWMBoundaryFromLabelImage\
  niftkCTEHighRes\
  niftkCTEHuttonLayering\
  niftkCTEJones2000\
  niftkCTEMaskedSmoothing\
  niftkCTEPrepareVolumes\
  niftkCTEYezzi2003\
  niftkCurveFitRegistration\
  niftkDecomposeAffineMatrix\
  niftkDeformationFieldTargetRegistrationError\
  niftkDicomSeriesReadImageWrite\
  niftkDilateMaskAndCrop\
  niftkDistanceTransform\
  niftkDoubleWindowBSI\
  niftkDynamicContrastEnhancementAnalysis\
  niftkExtractCurvatures\
  niftkExtractRegion\
  niftkExtractScalp\
  niftkExtractZeroCrossing\
  niftkFFD\
  niftkFillHoles\
  niftkForwardAndBackProjectionDifferenceFilter\
  niftkForwardProject3Dto2D\
  niftkGaussian\
  niftkGetMetricValue\
  niftkImageInfo\
  niftkImageReconstruction\
  niftkInject\
  niftkInvertAffineTransform\
  niftkInvertTransformation\
  niftkITKAffineResampleImage\
  niftkJacobianStatistics\
  niftkKMeansWindowBSI\
  niftkKMeansWindowWithLinearRegressionNormalisationBSI\
  niftkKNDoubleWindowBSI\
  niftkMammogramCharacteristics\
  niftkMTPDbc\
  niftkMultiplyTransformation\
  niftkMultiScaleHessianImageEnhancement2D\
  niftkMultiScaleHessianImageEnhancement3D\
  niftkOCTVolumeConstructor\
  niftkPadImage\
  niftkProjectionGeometry\
  niftkReorientateImage\
  niftkRescale\
  niftkResetDirectionField\
  niftkResetVoxelDimensionsField\
  niftkSampleImage\
  niftkSegmentationStatistics\
  niftkSeriesReadVolumeWrite\
  niftkSetBorderPixel\
  niftkShiftProb\
  niftkShiftScale\
  niftkShrinkImage\
  niftkSplitVolumeIntoVoxelPlanes\
  niftkSTAPLE\
  niftkSubtractSliceFromVolume\
  niftkSwapIntensity\
  niftkTestCompareImage\
  niftkTestImage\
  niftkThreshold\
  niftkTransformation\
  niftkTransformPoint3Dto2D\
  niftkVolToFreeSurfer\
  niftkVoxelWiseMaximumIntensities\
  niftkKmeansClassifier
do

  appname=${app:5}

echo "#/*============================================================================
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

NIFTK_CREATE_COMMAND_LINE_APPLICATION(
  NAME $app
  BUILD_CLI
  TARGET_LIBRARIES
    niftkcommon
    niftkITK
    niftkITKIO
    \${ITK_LIBRARIES}
    \${Boost_LIBRARIES}
)
" > $appname/CMakeLists.txt

done



for app in \
    niftkBreastMaskSegmentationFromMRI\
    niftkConvertImageToVTKStructuredGrid\
    niftkConvertNiftiVectorImage\
    niftkConvertPLYtoVTK\
    niftkDecimatePolyData\
    niftkGradientVectorField\
    niftkMapVolumeDataToPolyDataVertices\
    niftkMarchingCubes\
    niftkSmoothPolyData\
    niftkTransformPolyData
do

  appname=${app:5}

echo "#/*============================================================================
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

if(VTK_FOUND)

  NIFTK_CREATE_COMMAND_LINE_APPLICATION(
    NAME $app
    BUILD_CLI
    TARGET_LIBRARIES
      niftkVTK
      \${NIFTK_VTK_LIBS_BUT_WITHOUT_QT}
      niftkcommon
      niftkITK
      niftkITKIO
      \${ITK_LIBRARIES}
      \${Boost_LIBRARIES}
  )

endif()
" > $appname/CMakeLists.txt

done


for app in niftkMeshFromLabels
do

  appname=${app:5}

echo "#/*============================================================================
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

if(VTK_FOUND AND BUILD_MESHING)

  NIFTK_CREATE_COMMAND_LINE_APPLICATION(
    NAME $app
    BUILD_CLI
    TARGET_LIBRARIES
      niftkMeshing
      niftkITK
      niftkVTK
      niftkcommon
      \${ITK_LIBRARIES}
      \${CGAL_LIBRARIES}
      \${CGAL_3RD_PARTY_LIBRARIES}
  )

endif()
" > $appname/CMakeLists.txt

done




for app in niftkCreateBreastMesh
do

  appname=${app:5}

echo "#/*============================================================================
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

if(VTK_FOUND AND MITK_USE_Qt4)

  NIFTK_CREATE_COMMAND_LINE_APPLICATION(
    NAME $app
    BUILD_CLI
    TARGET_LIBRARIES
      \${NIFTK_VTK_LIBS_WITH_QT}
      \${QT_LIBRARIES}
      niftkVTK
      \${NIFTK_VTK_LIBS_BUT_WITHOUT_QT}
      niftkcommon
      niftkITK
      niftkITKIO
      \${ITK_LIBRARIES}
      \${Boost_LIBRARIES}
  )

endif()
" > $appname/CMakeLists.txt

done



for app in niftkCUDAInfo
do

  appname=${app:5}

echo "#/*============================================================================
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

if(CUDA_FOUND AND NIFTK_USE_CUDA)

  add_executable(${app} ${app}.cxx )

  include_directories(\${CUDA_TOOLKIT_INCLUDE})

  target_link_libraries(${app}
    PRIVATE
      \${CUDA_CUDA_LIBRARY}
      \${CUDA_CUDART_LIBRARY}
      niftkcommon
      niftkITK
      niftkITKIO
      \${ITK_LIBRARIES}
      \${Boost_LIBRARIES}
  )

  install(TARGETS ${app} RUNTIME DESTINATION \${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

endif()
" > $appname/CMakeLists.txt

done


for app in niftkAdd niftkOtsuThresholdImage niftkReorientateImage
do

  appname=${app:5}

echo "if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
" >> $appname/CMakeLists.txt

done

