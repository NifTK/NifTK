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
# Setup variables describing the library requirements.
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

###############################################################
# These programs use the new SlicerExecutionModel where
# we generate the command line parsing using TCLAP
###############################################################
IF(SlicerExecutionModel_FOUND)

  #-------------------------------------------------------------
  # Add new programs to this list.
  #-------------------------------------------------------------
  SET(APPS_DEPENDING_ON_SLICER_AND_APPLICATION_LINK_LIBRARIES
    niftkAdd
    niftkApplyMaskToImage
    niftkCreateMaskImage
    niftkCropImage
    niftkMultiply
    niftkSubtract
  )

  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
  
  FOREACH(APP ${APPS_DEPENDING_ON_SLICER_AND_APPLICATION_LINK_LIBRARIES})

    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/${APP}.xml.in ${CMAKE_CURRENT_BINARY_DIR}/${APP}.xml @ONLY )

    SEMMacroBuildNifTKCLI(
      NAME ${APP}
      EXECUTABLE_ONLY
      ${CMAKE_CURRENT_BINARY_DIR}
      TARGET_LIBRARIES ${NIFTK_APPLICATION_LINK_LIBRARIES}
    )
 
  ENDFOREACH()

  #-------------------------------------------------------------
  # Add new VTK programs which link to NiftyReg to this list.
  #-------------------------------------------------------------
  IF(VTK_FOUND)
    IF(NIFTYREG_FOUND)
      SET(VTK_APPS_DEPENDING_ON_SLICER_AND_NIFTYREG_LINK_LIBRARIES
	niftkF3DControlGridToVTKPolyData
	)

      if(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
	# OpenMP is disabled when the compiler is based on Clang
	find_package(OpenMP)
	if(OPENMP_FOUND)
          set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    	  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	endif(OPENMP_FOUND)
      endif(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
      
      ADD_DEFINITIONS(-DUSE_NIFTYREG)
      LINK_DIRECTORIES(${NIFTYREG_DIR}/lib)
      INCLUDE_DIRECTORIES(${NIFTYREG_DIR}/include)
      
      FOREACH(APP ${VTK_APPS_DEPENDING_ON_SLICER_AND_NIFTYREG_LINK_LIBRARIES})
	
	CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/${APP}.xml.in ${CMAKE_CURRENT_BINARY_DIR}/${APP}.xml @ONLY )
	
	SET(NIFTK_APPLICATION_WITH_VTK_AND_NIFTYREG_LINK_LIBRARIES
	  ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES} 
	  niftkITK
	  niftkVTK
	  _reg_KLdivergence
	  _reg_blockMatching
	  _reg_femTransformation
	  _reg_globalTransformation
	  _reg_localTransformation
	  _reg_maths
	  _reg_mutualinformation
	  _reg_resampling
	  _reg_ssd
	  _reg_tools
	  _reg_ReadWriteImage
	  reg_png
	  png
	  reg_nrrd
	  reg_NrrdIO
	  )
	
	SEMMacroBuildNifTKCLI(
	  NAME ${APP}
	  EXECUTABLE_ONLY
	  ${CMAKE_CURRENT_BINARY_DIR}
	  TARGET_LIBRARIES ${NIFTK_APPLICATION_WITH_VTK_AND_NIFTYREG_LINK_LIBRARIES}
	  )
 
      ENDFOREACH()
    ENDIF(NIFTYREG_FOUND)  
  ENDIF(VTK_FOUND)  
ENDIF()

###############################################################
# All other command line apps
###############################################################

##############################################################
# Add new programs to this list.
##############################################################
SET(APPS_DEPENDING_ON_NIFTK_APPLICATION_LINK_LIBRARIES
  niftkAbsImageFilter
  niftkAffine
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
  niftkContourExtractor2DImageFilter
  #niftkConvertDofToFsl
  niftkConvertImage
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
  niftkDilate
  niftkDilateMaskAndCrop
  niftkDistanceTransform
  niftkDoubleWindowBSI
  niftkErode
  niftkExtract2DSliceFrom3DImage
  niftkExtractCurvatures
  niftkExtractRegion
  niftkExtractScalp
  niftkExtractZeroCrossing
  niftkFFD
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
  niftkMTPDbc
  niftkMultiplyTransformation
  niftkMultiScaleHessianImageEnhancement2D
  niftkMultiScaleHessianImageEnhancement3D
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
  niftkSTAPLE
  niftkSubsampleImage
  niftkSubtractSliceFromVolume
  niftkSwapIntensity
  niftkTestCompareImage
  niftkTestImage
  niftkThreshold
  niftkTransformation
  niftkTransformPoint3Dto2D
  niftkVesselExtractor
  niftkVolToFreeSurfer
  niftkVoxelWiseMaximumIntensities
)

FOREACH(APP ${APPS_DEPENDING_ON_NIFTK_APPLICATION_LINK_LIBRARIES})
  ADD_EXECUTABLE(${APP} ${APP}.cxx )
  TARGET_LINK_LIBRARIES(${APP} ${NIFTK_APPLICATION_LINK_LIBRARIES} )
  MITK_INSTALL(TARGETS ${APP})
ENDFOREACH()
  
##############################################################
# Other command line applications, that require VTK.
##############################################################

IF(VTK_FOUND)
  
  ##############################################################
  # Add new programs to this list.
  ##############################################################
  SET(APPS_DEPENDING_ON_NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES
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

  FOREACH(APP ${APPS_DEPENDING_ON_NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES})
    ADD_EXECUTABLE(${APP} ${APP}.cxx )
    TARGET_LINK_LIBRARIES(${APP} ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES} )
    MITK_INSTALL(TARGETS ${APP})
  ENDFOREACH()

  IF (BUILD_MESHING) 
    IF (NOT WIN32) 
      SET(CMAKE_CXX_FLAGS "-L${CGAL_DIR}/..")
    ENDIF (NOT WIN32) 
    ADD_EXECUTABLE(niftkMeshFromLabels niftkMeshFromLabels.cxx)
    TARGET_LINK_LIBRARIES(niftkMeshFromLabels ${NIFTK_MESHING_APPLICATION_LIBRARIES})
    MITK_INSTALL(TARGETS niftkMeshFromLabels)
  ENDIF (BUILD_MESHING) 

ENDIF()

##############################################################
# Other command line applications, that require VTK and QT.
##############################################################
  
IF(VTK_FOUND)
  IF(QT_FOUND)

    ##############################################################
    # Add new programs to this list.
    ##############################################################      
    SET(APPS_DEPENDING_ON_NIFTK_APPLICATION_WITH_VTK_AND_QT_LINK_LIBRARIES
      niftkCreateBreastMesh
    )

    FOREACH(APP ${APPS_DEPENDING_ON_NIFTK_APPLICATION_WITH_VTK_AND_QT_LINK_LIBRARIES})
      ADD_EXECUTABLE(${APP} ${APP}.cxx )
      TARGET_LINK_LIBRARIES(${APP} ${NIFTK_APPLICATION_WITH_VTK_AND_QT_LINK_LIBRARIES} )
      MITK_INSTALL(TARGETS ${APP})
    ENDFOREACH()
  
  ENDIF(QT_FOUND)
ENDIF(VTK_FOUND)
  
##############################################################
# Other command line applications, that require CUDA.
##############################################################
  
IF(CUDA_FOUND)

  #INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCUDA.cmake)

  ##############################################################
  # Add new programs to this list.
  ##############################################################
  #SET(APPS_DEPENDING_ON_NIFTK_APPLICATION_WITH_CUDA_LINK_LIBRARIES
  #  niftkCUDAInfo
  #)
      
  #FOREACH(APP ${APPS_DEPENDING_ON_NIFTK_APPLICATION_WITH_CUDA_LINK_LIBRARIES})
  #  ADD_EXECUTABLE(${APP} ${APP}.cxx )
  #  TARGET_LINK_LIBRARIES(${APP} ${NIFTK_APPLICATION_WITH_CUDA_LINK_LIBRARIES} )
  #  INSTALL(TARGETS ${APP} RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  #ENDFOREACH()
  
ENDIF(CUDA_FOUND)
