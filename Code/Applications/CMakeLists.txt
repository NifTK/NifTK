#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#  
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2011-12-16 14:46:40 +0000 (Fri, 16 Dec 2011) $ 
#  Revision          : $Revision: 8050 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/
  
###############################################################
# This list of libraries serves as a default list for all apps.
###############################################################

SET(NIFTK_APPLICATION_LINK_LIBRARIES 
  niftkcommon 
  niftkITK
  ${NIFTK_ITK_LIB}
  ${Boost_LIBRARIES} 
)

SET(NIFTK_APPLICATION_WITH_CUDA_LINK_LIBRARIES
  niftkcudautilsgpu
  ${NIFTK_APPLICATION_LINK_LIBRARIES}
)

# Note: vtkRendering seems to force a dependency on Qt. See below.
SET(NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES
  ${NIFTK_VTK_LIBS_BUT_WITHOUT_QT}
  ${NIFTK_APPLICATION_LINK_LIBRARIES}
)

# Note: vtkRendering seems to force a dependency on Qt. See above.
SET(NIFTK_APPLICATION_WITH_VTK_AND_QT_LINK_LIBRARIES
  ${NIFTK_VTK_LIBS_WITH_QT}
  ${QT_LIBRARIES}
  ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES}
)

IF (BUILD_MESHING)
  SET(NIFTK_MESHING_APPLICATION_LIBRARIES
    niftkMeshingLib
    ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES}
    )
ENDIF (BUILD_MESHING)

##########################################################################
# Only build the following command line apps if BUILD_COMMAND_LINE is set.
##########################################################################
IF(BUILD_COMMAND_LINE_PROGRAMS)

  ##############################################################
  # Other command line applications.
  ##############################################################
  
  ADD_EXECUTABLE(niftkDilateMaskAndCrop niftkDilateMaskAndCrop.cxx )
  TARGET_LINK_LIBRARIES(niftkDilateMaskAndCrop ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkDilateMaskAndCrop RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkSubtract niftkSubtract.cxx )
  TARGET_LINK_LIBRARIES(niftkSubtract ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkSubtract RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkMultiply niftkMultiply.cxx )
  TARGET_LINK_LIBRARIES(niftkMultiply ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkMultiply RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkThreshold niftkThreshold.cxx )
  TARGET_LINK_LIBRARIES(niftkThreshold ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkThreshold RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkDilate niftkDilate.cxx )
  TARGET_LINK_LIBRARIES(niftkDilate ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkDilate RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkGaussian niftkGaussian.cxx )
  TARGET_LINK_LIBRARIES(niftkGaussian ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkGaussian RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkInject niftkInject.cxx )
  TARGET_LINK_LIBRARIES(niftkInject ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkInject RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCrop niftkCrop.cxx )
  TARGET_LINK_LIBRARIES(niftkCrop ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCrop RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkRescale niftkRescale.cxx )
  TARGET_LINK_LIBRARIES(niftkRescale ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkRescale RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkAffine niftkAffine.cxx )
  TARGET_LINK_LIBRARIES(niftkAffine ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkAffine RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkKMeansWindowBSI niftkKMeansWindowBSI.cxx )
  TARGET_LINK_LIBRARIES(niftkKMeansWindowBSI ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkKMeansWindowBSI RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCTEJones2000 niftkCTEJones2000.cxx )
  TARGET_LINK_LIBRARIES(niftkCTEJones2000 ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCTEJones2000 RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCTEYezzi2003 niftkCTEYezzi2003.cxx )
  TARGET_LINK_LIBRARIES(niftkCTEYezzi2003 ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCTEYezzi2003 RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkFFD niftkFFD.cxx )
  TARGET_LINK_LIBRARIES(niftkFFD ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkFFD RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkTestImage niftkTestImage.cxx )
  TARGET_LINK_LIBRARIES(niftkTestImage ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkTestImage RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkPadImage niftkPadImage.cxx )
  TARGET_LINK_LIBRARIES(niftkPadImage ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkPadImage RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCTEBourgeat2008 niftkCTEBourgeat2008.cxx )
  TARGET_LINK_LIBRARIES(niftkCTEBourgeat2008 ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCTEBourgeat2008 RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkBlockMatching niftkBlockMatching.cxx )
  TARGET_LINK_LIBRARIES(niftkBlockMatching ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkBlockMatching RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkTransformation niftkTransformation.cxx )
  TARGET_LINK_LIBRARIES(niftkTransformation ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkTransformation RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkITKAffineResampleImage niftkITKAffineResampleImage.cxx )
  TARGET_LINK_LIBRARIES(niftkITKAffineResampleImage ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkITKAffineResampleImage RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkAbsImageFilter niftkAbsImageFilter.cxx )
  TARGET_LINK_LIBRARIES(niftkAbsImageFilter ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkAbsImageFilter RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkSTAPLE niftkSTAPLE.cxx )
  TARGET_LINK_LIBRARIES(niftkSTAPLE ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkSTAPLE RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkJacobianStatistics niftkJacobianStatistics.cxx )
  TARGET_LINK_LIBRARIES(niftkJacobianStatistics ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkJacobianStatistics RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkConvertImage niftkConvertImage.cxx )
  TARGET_LINK_LIBRARIES(niftkConvertImage ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkConvertImage RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkConvertToMidasStr niftkConvertToMidasStr.cxx )
  TARGET_LINK_LIBRARIES(niftkConvertToMidasStr ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkConvertToMidasStr RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkConvertMidasStrToNii niftkConvertMidasStrToNii.cxx )
  TARGET_LINK_LIBRARIES(niftkConvertMidasStrToNii ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkConvertMidasStrToNii RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkSegmentationStatistics niftkSegmentationStatistics.cxx )
  TARGET_LINK_LIBRARIES(niftkSegmentationStatistics ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkSegmentationStatistics RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkConvertTransformToRIREFormat niftkConvertTransformToRIREFormat.cxx )
  TARGET_LINK_LIBRARIES(niftkConvertTransformToRIREFormat ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkConvertTransformToRIREFormat RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkShiftScale niftkShiftScale.cxx )
  TARGET_LINK_LIBRARIES(niftkShiftScale ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkShiftScale RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkExtract2DSliceFrom3DImage niftkExtract2DSliceFrom3DImage.cxx )
  TARGET_LINK_LIBRARIES(niftkExtract2DSliceFrom3DImage ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkExtract2DSliceFrom3DImage RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkDoubleWindowBSI niftkDoubleWindowBSI.cxx )
  TARGET_LINK_LIBRARIES(niftkDoubleWindowBSI ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkDoubleWindowBSI RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkKNDoubleWindowBSI niftkKNDoubleWindowBSI.cxx )
  TARGET_LINK_LIBRARIES(niftkKNDoubleWindowBSI ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkKNDoubleWindowBSI RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkKMeansWindowWithLinearRegressionNormalisationBSI niftkKMeansWindowWithLinearRegressionNormalisationBSI.cxx )
  TARGET_LINK_LIBRARIES(niftkKMeansWindowWithLinearRegressionNormalisationBSI ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkKMeansWindowWithLinearRegressionNormalisationBSI RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkBSI niftkBSI.cxx )
  TARGET_LINK_LIBRARIES(niftkBSI ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkBSI RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkForwardProject3Dto2D niftkForwardProject3Dto2D.cxx )
  TARGET_LINK_LIBRARIES(niftkForwardProject3Dto2D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkForwardProject3Dto2D RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkBackProject2Dto3D niftkBackProject2Dto3D.cxx )
  TARGET_LINK_LIBRARIES(niftkBackProject2Dto3D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkBackProject2Dto3D RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCTEDas2009 niftkCTEDas2009.cxx )
  TARGET_LINK_LIBRARIES(niftkCTEDas2009 ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCTEDas2009 RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCombineSegmentations niftkCombineSegmentations.cxx )
  TARGET_LINK_LIBRARIES(niftkCombineSegmentations ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCombineSegmentations RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCTEPrepareVolumes niftkCTEPrepareVolumes.cxx )
  TARGET_LINK_LIBRARIES(niftkCTEPrepareVolumes ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCTEPrepareVolumes RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkImageInfo niftkImageInfo.cxx )
  TARGET_LINK_LIBRARIES(niftkImageInfo ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkImageInfo RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkBilateralImageFilter niftkBilateralImageFilter.cxx )
  TARGET_LINK_LIBRARIES(niftkBilateralImageFilter ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkBilateralImageFilter RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkAdd niftkAdd.cxx )
  TARGET_LINK_LIBRARIES(niftkAdd ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkAdd RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkSwapIntensity niftkSwapIntensity.cxx )
  TARGET_LINK_LIBRARIES(niftkSwapIntensity ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkSwapIntensity RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkVolToFreeSurfer niftkVolToFreeSurfer.cxx )
  TARGET_LINK_LIBRARIES(niftkVolToFreeSurfer ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkVolToFreeSurfer RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCreateAffineTransform niftkCreateAffineTransform.cxx )
  TARGET_LINK_LIBRARIES(niftkCreateAffineTransform ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCreateAffineTransform RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkProjectionGeometry niftkProjectionGeometry.cxx )
  TARGET_LINK_LIBRARIES(niftkProjectionGeometry ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkProjectionGeometry RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkSubtractSliceFromVolume niftkSubtractSliceFromVolume.cxx )
  TARGET_LINK_LIBRARIES(niftkSubtractSliceFromVolume ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkSubtractSliceFromVolume RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkResetDirectionField niftkResetDirectionField.cxx )
  TARGET_LINK_LIBRARIES(niftkResetDirectionField ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkResetDirectionField RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkForwardAndBackProjectionDifferenceFilter niftkForwardAndBackProjectionDifferenceFilter.cxx )
  TARGET_LINK_LIBRARIES(niftkForwardAndBackProjectionDifferenceFilter ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkForwardAndBackProjectionDifferenceFilter RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkImageReconstruction niftkImageReconstruction.cxx )
  TARGET_LINK_LIBRARIES(niftkImageReconstruction ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkImageReconstruction RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkShrinkImage niftkShrinkImage.cxx )
  TARGET_LINK_LIBRARIES(niftkShrinkImage ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkShrinkImage RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCTEHighRes niftkCTEHighRes.cxx )
  TARGET_LINK_LIBRARIES(niftkCTEHighRes ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCTEHighRes RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkAtlasStatistics niftkAtlasStatistics.cxx )
  TARGET_LINK_LIBRARIES(niftkAtlasStatistics ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkAtlasStatistics RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCTEAcosta2009Subsampling niftkCTEAcosta2009Subsampling.cxx )
  TARGET_LINK_LIBRARIES(niftkCTEAcosta2009Subsampling ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCTEAcosta2009Subsampling RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCTEHuttonLayering niftkCTEHuttonLayering.cxx )
  TARGET_LINK_LIBRARIES(niftkCTEHuttonLayering ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCTEHuttonLayering RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkAnd niftkAnd.cxx )
  TARGET_LINK_LIBRARIES(niftkAnd ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkAnd RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkExtractRegion niftkExtractRegion.cxx )
  TARGET_LINK_LIBRARIES(niftkExtractRegion ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkExtractRegion RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkDecomposeAffineMatrix niftkDecomposeAffineMatrix.cxx )
  TARGET_LINK_LIBRARIES(niftkDecomposeAffineMatrix ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkDecomposeAffineMatrix RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCreateMaskFromLabels niftkCreateMaskFromLabels.cxx )
  TARGET_LINK_LIBRARIES(niftkCreateMaskFromLabels ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCreateMaskFromLabels RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkAverage niftkAverage.cxx )
  TARGET_LINK_LIBRARIES(niftkAverage ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkAverage RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkInvertAffineTransform niftkInvertAffineTransform.cxx )
  TARGET_LINK_LIBRARIES(niftkInvertAffineTransform ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkInvertAffineTransform RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkGetMetricValue niftkGetMetricValue.cxx )
  TARGET_LINK_LIBRARIES(niftkGetMetricValue ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkGetMetricValue RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkConnectedComponents niftkConnectedComponents.cxx )
  TARGET_LINK_LIBRARIES(niftkConnectedComponents ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkConnectedComponents RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkResetVoxelDimensionsField niftkResetVoxelDimensionsField.cxx )
  TARGET_LINK_LIBRARIES(niftkResetVoxelDimensionsField ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkResetVoxelDimensionsField RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkErode niftkErode )
  TARGET_LINK_LIBRARIES(niftkErode ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkErode RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCreateTransformation niftkCreateTransformation.cxx )
  TARGET_LINK_LIBRARIES(niftkCreateTransformation ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCreateTransformation RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkDeformationFieldTargetRegistrationError niftkDeformationFieldTargetRegistrationError.cxx )
  TARGET_LINK_LIBRARIES(niftkDeformationFieldTargetRegistrationError ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkDeformationFieldTargetRegistrationError RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkSetBorderPixel niftkSetBorderPixel.cxx )
  TARGET_LINK_LIBRARIES(niftkSetBorderPixel ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkSetBorderPixel RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  #Need to work on it more later. 
  #ADD_EXECUTABLE(niftkConvertDofToFsl niftkConvertDofToFsl.cxx )
  #TARGET_LINK_LIBRARIES(niftkConvertDofToFsl ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  #INSTALL(TARGETS niftkConvertDofToFsl RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkFillHoles niftkFillHoles.cxx )
  TARGET_LINK_LIBRARIES(niftkFillHoles ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkFillHoles RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkShiftProb niftkShiftProb.cxx)
  TARGET_LINK_LIBRARIES(niftkShiftProb ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkShiftProb RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkExtractCurvatures niftkExtractCurvatures)
  TARGET_LINK_LIBRARIES(niftkExtractCurvatures ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkExtractCurvatures RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkMultiScaleHessianImageEnhancement2D niftkMultiScaleHessianImageEnhancement2D.cxx )
  TARGET_LINK_LIBRARIES(niftkMultiScaleHessianImageEnhancement2D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkMultiScaleHessianImageEnhancement2D RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkMultiScaleHessianImageEnhancement3D niftkMultiScaleHessianImageEnhancement3D.cxx )
  TARGET_LINK_LIBRARIES(niftkMultiScaleHessianImageEnhancement3D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkMultiScaleHessianImageEnhancement3D RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkExtractZeroCrossing niftkExtractZeroCrossing )
  TARGET_LINK_LIBRARIES(niftkExtractZeroCrossing ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkExtractZeroCrossing RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkDistanceTransform niftkDistanceTransform )
  TARGET_LINK_LIBRARIES(niftkDistanceTransform ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkDistanceTransform RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCTEMaskedSmoothing niftkCTEMaskedSmoothing )
  TARGET_LINK_LIBRARIES(niftkCTEMaskedSmoothing ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCTEMaskedSmoothing RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCTEExtractGMWMBoundaryFromLabelImage niftkCTEExtractGMWMBoundaryFromLabelImage )
  TARGET_LINK_LIBRARIES(niftkCTEExtractGMWMBoundaryFromLabelImage ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCTEExtractGMWMBoundaryFromLabelImage RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkCTEAssignAtlasValues niftkCTEAssignAtlasValues )
  TARGET_LINK_LIBRARIES(niftkCTEAssignAtlasValues ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkCTEAssignAtlasValues RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkTransformPoint3Dto2D niftkTransformPoint3Dto2D.cxx )
  TARGET_LINK_LIBRARIES(niftkTransformPoint3Dto2D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkTransformPoint3Dto2D RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkSubsampleImage niftkSubsampleImage.cxx )
  TARGET_LINK_LIBRARIES(niftkSubsampleImage ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkSubsampleImage RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkDicomSeriesReadImageWrite niftkDicomSeriesReadImageWrite.cxx )
  TARGET_LINK_LIBRARIES(niftkDicomSeriesReadImageWrite ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkDicomSeriesReadImageWrite RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  #ADD_EXECUTABLE(niftkExtractScalp niftkExtractScalp.cxx )
  #TARGET_LINK_LIBRARIES(niftkExtractScalp ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  #INSTALL(TARGETS niftkExtractScalp RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkMTPDbc niftkMTPDbc.cxx)
  TARGET_LINK_LIBRARIES(niftkMTPDbc ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkMTPDbc RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkConvertNiftiVectorImage niftkConvertNiftiVectorImage.cxx )
  TARGET_LINK_LIBRARIES(niftkConvertNiftiVectorImage ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkConvertNiftiVectorImage RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkContourExtractor2DImageFilter niftkContourExtractor2DImageFilter.cxx )
  TARGET_LINK_LIBRARIES(niftkContourExtractor2DImageFilter ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkContourExtractor2DImageFilter RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkInvertTransformation niftkInvertTransformation.cxx )
  TARGET_LINK_LIBRARIES(niftkInvertTransformation ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkInvertTransformation RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkComputeMeanTransformation niftkComputeMeanTransformation.cxx )
  TARGET_LINK_LIBRARIES(niftkComputeMeanTransformation ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkComputeMeanTransformation RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ADD_EXECUTABLE(niftkMultiplyTransformation niftkMultiplyTransformation.cxx )
  TARGET_LINK_LIBRARIES(niftkMultiplyTransformation ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkMultiplyTransformation RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

  ADD_EXECUTABLE(niftkComposeITKAffineTransformations niftkComposeITKAffineTransformations.cxx )
  TARGET_LINK_LIBRARIES(niftkComposeITKAffineTransformations ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkComposeITKAffineTransformations RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

  ADD_EXECUTABLE(niftkTestCompareImage niftkTestCompareImage.cxx )
  TARGET_LINK_LIBRARIES(niftkTestCompareImage ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkTestCompareImage RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ##############################################################
  # Other command line applications, that require VTK.
  ##############################################################
  
  IF(VTK_FOUND)
  
    ADD_EXECUTABLE(niftkGradientVectorField niftkGradientVectorField )
    TARGET_LINK_LIBRARIES(niftkGradientVectorField ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES} )
    INSTALL(TARGETS niftkGradientVectorField RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
      
    ADD_EXECUTABLE(niftkMarchingCubes niftkMarchingCubes )
    TARGET_LINK_LIBRARIES(niftkMarchingCubes ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES} )
    INSTALL(TARGETS niftkMarchingCubes RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
    ADD_EXECUTABLE(niftkMapVolumeDataToPolyDataVertices niftkMapVolumeDataToPolyDataVertices )
    TARGET_LINK_LIBRARIES(niftkMapVolumeDataToPolyDataVertices ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES} )
    INSTALL(TARGETS niftkMapVolumeDataToPolyDataVertices RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
      
    ADD_EXECUTABLE(niftkDecimatePolyData niftkDecimatePolyData.cxx )
    TARGET_LINK_LIBRARIES(niftkDecimatePolyData ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES} )
    INSTALL(TARGETS niftkDecimatePolyData RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
    ADD_EXECUTABLE(niftkSmoothPolyData niftkSmoothPolyData.cxx )
    TARGET_LINK_LIBRARIES(niftkSmoothPolyData ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES} )
    INSTALL(TARGETS niftkSmoothPolyData RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
    ADD_EXECUTABLE(niftkTransformPolyData niftkTransformPolyData.cxx )
    TARGET_LINK_LIBRARIES(niftkTransformPolyData ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES} )
    INSTALL(TARGETS niftkTransformPolyData RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
    ADD_EXECUTABLE(niftkConvertImageToVTKStructuredGrid niftkConvertImageToVTKStructuredGrid.cxx )
    TARGET_LINK_LIBRARIES(niftkConvertImageToVTKStructuredGrid ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES} )
    INSTALL(TARGETS niftkConvertImageToVTKStructuredGrid RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
   
  
    IF (BUILD_MESHING) 
      IF (NOT WIN32) 
	      SET(CMAKE_CXX_FLAGS "-L${CGAL_DIR}/..")
      ENDIF (NOT WIN32) 
      ADD_EXECUTABLE(niftkMeshFromLabels niftkMeshFromLabels.cxx)
      TARGET_LINK_LIBRARIES(niftkMeshFromLabels ${NIFTK_MESHING_APPLICATION_LIBRARIES})
      INSTALL(TARGETS niftkMeshFromLabels RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
    ENDIF (BUILD_MESHING) 
  
  ENDIF(VTK_FOUND)
  
  ##############################################################
  # Other command line applications, that require VTK and QT.
  ##############################################################
  
  IF(VTK_FOUND)
    IF(QT_FOUND)
      
      ADD_EXECUTABLE(niftkCreateBreastMesh niftkCreateBreastMesh.cxx )
      TARGET_LINK_LIBRARIES(niftkCreateBreastMesh ${NIFTK_APPLICATION_WITH_VTK_AND_QT_LINK_LIBRARIES} )
      INSTALL(TARGETS niftkCreateBreastMesh RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
    ENDIF(QT_FOUND)
  ENDIF(VTK_FOUND)
  
  ##############################################################
  # Other command line applications, that require CUDA.
  ##############################################################
  
  IF(CUDA_FOUND)
  
    #INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCUDA.cmake)
    
    #CUDA_ADD_EXECUTABLE(niftkCUDAInfo niftkCUDAInfo.cxx )
    #TARGET_LINK_LIBRARIES(niftkCUDAInfo ${NIFTK_APPLICATION_WITH_CUDA_LINK_LIBRARIES} )
    #INSTALL(TARGETS niftkCUDAInfo RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  
  ENDIF(CUDA_FOUND)

##################################################################################
# Don't forget we are inside an if block, only building command line if necessary.
##################################################################################
ENDIF(BUILD_COMMAND_LINE_PROGRAMS)

