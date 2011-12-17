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
#  Last Changed      : $LastChangedDate: 2011-10-17 12:41:50 +0100 (Mon, 17 Oct 2011) $ 
#  Revision          : $Revision: 7533 $
#  Last modified by  : $Author: jhh $
#
#  Original author   : j.hipwell@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

# This is the root ITK CMakeLists file.
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)

SET(NIFTK_APPLICATION_LINK_LIBRARIES 
  niftkcommon 
  niftkITK
  ITKCommon 
  ITKIO 
  ITKBasicFilters
  ITKAlgorithms
  itkzlib
  ${Boost_LIBRARIES} 
  )

# Find ITK.
FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})


###########################################################################
# ITK Applications
###########################################################################

ADD_EXECUTABLE(niftkCommandLineParserExample niftkCommandLineParserExample.cxx )
TARGET_LINK_LIBRARIES(niftkCommandLineParserExample ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkCommandLineParserExample RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkAffineRegistrationAndReconstruction niftkAffineRegistrationAndReconstruction.cxx )
TARGET_LINK_LIBRARIES(niftkAffineRegistrationAndReconstruction ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkAffineRegistrationAndReconstruction RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkPCADeformationModelRegistration niftkPCADeformationModelRegistration.cxx )
TARGET_LINK_LIBRARIES(niftkPCADeformationModelRegistration ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkPCADeformationModelRegistration RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkPCADeformationModelTransformation niftkPCADeformationModelTransformation.cxx )
TARGET_LINK_LIBRARIES(niftkPCADeformationModelTransformation ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkPCADeformationModelTransformation RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkPCADeformationModelEstimator niftkPCADeformationModelEstimator.cxx )
TARGET_LINK_LIBRARIES(niftkPCADeformationModelEstimator ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkPCADeformationModelEstimator RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkIterativePCARegistrationAndReconstruction niftkIterativePCARegistrationAndReconstruction.cxx )
TARGET_LINK_LIBRARIES(niftkIterativePCARegistrationAndReconstruction ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkIterativePCARegistrationAndReconstruction RUNTIME DESTINATION bin COMPONENT applications)

#ADD_EXECUTABLE(niftkFitPCADeformationModelToDeformationField niftkFitPCADeformationModelToDeformationField.cxx )
#TARGET_LINK_LIBRARIES(niftkFitPCADeformationModelToDeformationField ${NIFTK_APPLICATION_LINK_LIBRARIES} )
#INSTALL(TARGETS niftkFitPCADeformationModelToDeformationField RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkSimulateMammogramFromMRI niftkSimulateMammogramFromMRI.cxx )
TARGET_LINK_LIBRARIES(niftkSimulateMammogramFromMRI ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkSimulateMammogramFromMRI RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkDeformationVectorFieldToDisplacementComponentImages niftkDeformationVectorFieldToDisplacementComponentImages.cxx )
TARGET_LINK_LIBRARIES(niftkDeformationVectorFieldToDisplacementComponentImages ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkDeformationVectorFieldToDisplacementComponentImages RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkTestImageAndArray niftkTestImageAndArray.cxx )
TARGET_LINK_LIBRARIES(niftkTestImageAndArray ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkTestImageAndArray RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(itkMultiScaleHessianBasedMeasureImageFilterTest itkMultiScaleHessianBasedMeasureImageFilterTest.cxx )
TARGET_LINK_LIBRARIES(itkMultiScaleHessianBasedMeasureImageFilterTest ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS itkMultiScaleHessianBasedMeasureImageFilterTest RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkMultiScaleRaundahlStripinessEnhancement2D niftkMultiScaleRaundahlStripinessEnhancement2D.cxx )
TARGET_LINK_LIBRARIES(niftkMultiScaleRaundahlStripinessEnhancement2D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkMultiScaleRaundahlStripinessEnhancement2D RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkHessianEigenAnalysis2D niftkHessianEigenAnalysis2D.cxx )
TARGET_LINK_LIBRARIES(niftkHessianEigenAnalysis2D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkHessianEigenAnalysis2D RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkBasicImageFeatures niftkBasicImageFeatures.cxx )
TARGET_LINK_LIBRARIES(niftkBasicImageFeatures ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkBasicImageFeatures RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkOtsuThresholdImageFilter niftkOtsuThresholdImageFilter.cxx )
TARGET_LINK_LIBRARIES(niftkOtsuThresholdImageFilter ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkOtsuThresholdImageFilter RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkDICOMMammoConvert niftkDICOMMammoConvert.cxx )
TARGET_LINK_LIBRARIES(niftkDICOMMammoConvert ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkDICOMMammoConvert RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkScalarToRGB niftkScalarToRGB.cxx )
TARGET_LINK_LIBRARIES(niftkScalarToRGB ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkScalarToRGB RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkSplatImage3Dto2D niftkSplatImage3Dto2D.cxx )
TARGET_LINK_LIBRARIES(niftkSplatImage3Dto2D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkSplatImage3Dto2D RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkGroupwiseAffineRegistration niftkGroupwiseAffineRegistration.cxx )
TARGET_LINK_LIBRARIES(niftkGroupwiseAffineRegistration ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkGroupwiseAffineRegistration RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkMeanVoxelwiseIntensityOfMultipleImages niftkMeanVoxelwiseIntensityOfMultipleImages.cxx )
TARGET_LINK_LIBRARIES(niftkMeanVoxelwiseIntensityOfMultipleImages ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkMeanVoxelwiseIntensityOfMultipleImages RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkReadXMLTest niftkReadXMLTest.cxx )
TARGET_LINK_LIBRARIES(niftkReadXMLTest ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkReadXMLTest RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkTransformPoint niftkTransformPoint.cxx )
TARGET_LINK_LIBRARIES(niftkTransformPoint ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkTransformPoint RUNTIME DESTINATION bin COMPONENT applications)

# ADD_EXECUTABLE(niftkAtlasAffineMammoSegmentation niftkAtlasAffineMammoSegmentation.cxx )
# TARGET_LINK_LIBRARIES(niftkAtlasAffineMammoSegmentation ${NIFTK_APPLICATION_LINK_LIBRARIES} )
# INSTALL(TARGETS niftkAtlasAffineMammoSegmentation RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkGoodbyeCruelWorld niftkGoodbyeCruelWorld.cxx )
TARGET_LINK_LIBRARIES(niftkGoodbyeCruelWorld ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkGoodbyeCruelWorld RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkComputeAffineRegistrationError niftkComputeAffineRegistrationError.cxx )
TARGET_LINK_LIBRARIES(niftkComputeAffineRegistrationError ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkComputeAffineRegistrationError RUNTIME DESTINATION bin COMPONENT applications)

#ADD_EXECUTABLE(argyriosTesting argyriosTesting.cxx )
#TARGET_LINK_LIBRARIES(argyriosTesting ${NIFTK_APPLICATION_LINK_LIBRARIES} )
#INSTALL(TARGETS argyriosTesting RUNTIME DESTINATION bin COMPONENT applications)

#ADD_EXECUTABLE(landmark_registration3D landmark_registration3D.cxx )
#TARGET_LINK_LIBRARIES(landmark_registration3D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
#INSTALL(TARGETS landmark_registration3D RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkBreastDicomSeriesReadImageWrite niftkBreastDicomSeriesReadImageWrite.cxx )
TARGET_LINK_LIBRARIES(niftkBreastDicomSeriesReadImageWrite ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkBreastDicomSeriesReadImageWrite RUNTIME DESTINATION bin COMPONENT applications)

#ADD_EXECUTABLE(niftkComputeLandmarkAffineRegistrationError2D3D niftkComputeLandmarkAffineRegistrationError2D3D.cxx )
#TARGET_LINK_LIBRARIES(niftkComputeLandmarkAffineRegistrationError2D3D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
#INSTALL(TARGETS niftkComputeLandmarkAffineRegistrationError2D3D RUNTIME DESTINATION bin COMPONENT applications)

#ADD_EXECUTABLE(niftkMammoConvert niftkMammoConvert.cxx )
#TARGET_LINK_LIBRARIES(niftkMammoConvert ${NIFTK_APPLICATION_LINK_LIBRARIES} )
#INSTALL(TARGETS niftkMammoConvert RUNTIME DESTINATION bin COMPONENT applications)



################################################################################################
# Command line applications that require VTK (and vtkRendering requires Qt on some platforms???)
################################################################################################

IF(VTK_FOUND)
  IF(QT_FOUND)

    SET(NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES 
      niftkcommon 
      niftkITK
      ITKCommon 
      ITKIO 
      ITKBasicFilters
      ITKAlgorithms
      vtkIO
      vtkCommon
      vtkGraphics
      vtkImaging
      vtkVolumeRendering
      vtkRendering
      itkzlib
      ${Boost_LIBRARIES}
      ${QT_LIBRARIES} 
    )

    ADD_EXECUTABLE(niftkVTKVoxelModeller niftkVTKVoxelModeller.cxx )
    TARGET_LINK_LIBRARIES(niftkVTKVoxelModeller ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES} )
    INSTALL(TARGETS niftkVTKVoxelModeller RUNTIME DESTINATION bin COMPONENT applications)

    ADD_EXECUTABLE(niftkVTKVolumeRender niftkVTKVolumeRender.cxx )
    TARGET_LINK_LIBRARIES(niftkVTKVolumeRender ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES} )
    INSTALL(TARGETS niftkVTKVolumeRender RUNTIME DESTINATION bin COMPONENT applications)

  #  ADD_EXECUTABLE(niftkVTKInvertTransformation niftkVTKInvertTransformation.cxx )
  #  TARGET_LINK_LIBRARIES(niftkVTKInvertTransformation ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES} )
  #  INSTALL(TARGETS niftkVTKInvertTransformation RUNTIME DESTINATION bin COMPONENT applications)

  ENDIF(QT_FOUND)
ENDIF(VTK_FOUND)


##############################################################
# Other command line applications, that require CUDA.
##############################################################

# Find CUDA
IF(USE_CUDA)
  FIND_PACKAGE(CUDA)
  IF(CUDA_FOUND)
    INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})
    CUDA_INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})
    ADD_DEFINITIONS(-D_USE_CUDA)
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D_USE_CUDA")
  ELSE(CUDA_FOUND)
    MESSAGE("Didn't find CUDA")
  ENDIF(CUDA_FOUND)
ENDIF(USE_CUDA)

IF(CUDA_FOUND)

  INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCUDA.cmake)

  INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE})
  INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})
  CUDA_INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})

  INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/C/common/inc ${CUDA_SDK_ROOT_DIR}/shared/inc)
  LINK_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/C/common/lib ${CUDA_SDK_ROOT_DIR}/shared/lib)
  
  SET(NIFTK_APPLICATION_WITH_CUDA_LINK_LIBRARIES
    niftkcommon
    niftkITK
    niftkcudautilsgpu 
    niftkCUDAProjectionToolbox
    ${Boost_LIBRARIES} 
  )

  #CUDA_ADD_EXECUTABLE( niftkRayCastUsingCUDA niftkRayCastUsingCUDA.cxx )
  #TARGET_LINK_LIBRARIES( niftkRayCastUsingCUDA ${NIFTK_APPLICATION_WITH_CUDA_LINK_LIBRARIES} )
  #INSTALL(TARGETS niftkRayCastUsingCUDA RUNTIME DESTINATION bin COMPONENT applications)


  ###########################################################################
  # Programs that link to NiftySim
  ###########################################################################

  IF(NIFTK_USE_NIFTYSIM)

    INCLUDE(${CMAKE_SOURCE_DIR}/CMake/FindNiftySim.cmake)

    INCLUDE_DIRECTORIES(${NIFTYSIM_INCLUDE_DIR})
    LINK_DIRECTORIES(${NIFTYSIM_DIR}/lib /opt/fontconfig/lib/)

    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D_GPU_")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GPU_")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
  
    SET(NIFTK_APPLICATION_WITH_NIFTYSIM_LINK_LIBRARIES
      niftkcommon
      niftkITK
      niftkcudautilsgpu 
      ${Boost_LIBRARIES} 
      tled
      xmlParser
      )
  
    IF(VTK_FOUND)
      IF(QT_FOUND)

	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_Visualisation_")
      
	SET(NIFTK_APPLICATION_WITH_NIFTYSIM_LINK_LIBRARIES 
	  ${NIFTK_APPLICATION_WITH_NIFTYSIM_LINK_LIBRARIES}
	  viz
	  niftkITKVTK
	  niftkVTK
	  vtkHybrid
	  vtkRendering
	  vtkGraphics
	  vtkImaging
	  vtkIO
	  vtkCommon
	  vtkVolumeRendering
	  QVTK
	  ${QT_LIBRARIES} 
	  )
      
      ENDIF(QT_FOUND)
    ENDIF(VTK_FOUND)


    ADD_EXECUTABLE(niftkNiftySimTransformation niftkNiftySimTransformation.cxx )
    TARGET_LINK_LIBRARIES(niftkNiftySimTransformation ${NIFTK_APPLICATION_WITH_NIFTYSIM_LINK_LIBRARIES} )
    INSTALL(TARGETS niftkNiftySimTransformation RUNTIME DESTINATION bin COMPONENT applications)

    ADD_EXECUTABLE(niftkNiftySimPlateCompressionRegistration niftkNiftySimPlateCompressionRegistration.cxx )
    TARGET_LINK_LIBRARIES(niftkNiftySimPlateCompressionRegistration ${NIFTK_APPLICATION_WITH_NIFTYSIM_LINK_LIBRARIES} )
    INSTALL(TARGETS niftkNiftySimPlateCompressionRegistration RUNTIME DESTINATION bin COMPONENT applications)

  ENDIF(NIFTK_USE_NIFTYSIM)

ENDIF(CUDA_FOUND)

