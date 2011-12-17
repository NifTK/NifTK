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
#  Last Changed      : $LastChangedDate: $
#  Revision          : $Revision: $
#  Last modified by  : $Author: $
#
#  Original author   : t.mertzanidou@cs.ucl.ac.uk
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
  ${Log4cplus_LIB}
  ${Boost_LIBRARIES} 
  )

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
  itkzlib
  ${Log4cplus_LIB}
  ${Boost_LIBRARIES} 
  )

# Find ITK.
FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

SET( CMAKE_BUILD_TYPE Debug )

ADD_EXECUTABLE(lesionCorridorDistance lesionCorridorDistance.cpp)
TARGET_LINK_LIBRARIES(lesionCorridorDistance ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS lesionCorridorDistance RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(binaryDilateImage binaryDilateImage.cpp)
TARGET_LINK_LIBRARIES(binaryDilateImage ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS binaryDilateImage RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(translateVolume translateVolume.cxx )
TARGET_LINK_LIBRARIES(translateVolume ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS translateVolume RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(fromGIPLStoMHA fromGIPLStoMHA.cpp )
TARGET_LINK_LIBRARIES(fromGIPLStoMHA ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS fromGIPLStoMHA RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(fromMHAtoGIPL fromMHAtoGIPL.cpp )
TARGET_LINK_LIBRARIES(fromMHAtoGIPL ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS fromMHAtoGIPL RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(combineCTandSpike combineCTandSpike.cpp )
TARGET_LINK_LIBRARIES(combineCTandSpike ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS combineCTandSpike RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(combineMaskedVols combineMaskedVols.cpp )
TARGET_LINK_LIBRARIES(combineMaskedVols ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS combineMaskedVols RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(inverseTranslateMammosForReg inverseTranslateMammosForReg.cpp )
TARGET_LINK_LIBRARIES(inverseTranslateMammosForReg ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS inverseTranslateMammosForReg RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(invertIntensities invertIntensities.cpp )
TARGET_LINK_LIBRARIES(invertIntensities ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS invertIntensities RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkThresholdImage niftkThresholdImage.cpp )
TARGET_LINK_LIBRARIES(niftkThresholdImage ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkThresholdImage RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(computeBreastDensity computeBreastDensity.cpp )
TARGET_LINK_LIBRARIES(computeBreastDensity ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS computeBreastDensity RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(computeMaskVolume computeMaskVolume.cpp )
TARGET_LINK_LIBRARIES(computeMaskVolume ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS computeMaskVolume RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(pointMatch pointMatch.cpp )
TARGET_LINK_LIBRARIES(pointMatch ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS pointMatch RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(computeNCC computeNCC.cpp )
TARGET_LINK_LIBRARIES(computeNCC ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS computeNCC RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkAffineRegistration2D3D niftkAffineRegistration2D3D.cxx )
TARGET_LINK_LIBRARIES(niftkAffineRegistration2D3D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkAffineRegistration2D3D RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkAffineDRR niftkAffineDRR.cxx )
TARGET_LINK_LIBRARIES(niftkAffineDRR ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkAffineDRR RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkAffineTransform2D3D niftkAffineTransform2D3D.cxx )
TARGET_LINK_LIBRARIES(niftkAffineTransform2D3D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkAffineTransform2D3D RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkAffineInvDirectionReg2D3D niftkAffineInvDirectionReg2D3D.cxx )
TARGET_LINK_LIBRARIES(niftkAffineInvDirectionReg2D3D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkAffineInvDirectionReg2D3D RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkPCADeformationModelRegistration2D3D niftkPCADeformationModelRegistration2D3D.cxx )
TARGET_LINK_LIBRARIES(niftkPCADeformationModelRegistration2D3D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkPCADeformationModelRegistration2D3D RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkRigidAndPCADeformationModelRegistration2D3D niftkRigidAndPCADeformationModelRegistration2D3D.cxx )
TARGET_LINK_LIBRARIES(niftkRigidAndPCADeformationModelRegistration2D3D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkRigidAndPCADeformationModelRegistration2D3D RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkRigidAndPCADeformationModelRegistrationWithMask2D3D niftkRigidAndPCADeformationModelRegistrationWithMask2D3D.cxx )
TARGET_LINK_LIBRARIES(niftkRigidAndPCADeformationModelRegistrationWithMask2D3D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkRigidAndPCADeformationModelRegistrationWithMask2D3D RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkRigidAndPCADeformationModelRegistrationMaskCog niftkRigidAndPCADeformationModelRegistrationMaskCog.cxx )
TARGET_LINK_LIBRARIES(niftkRigidAndPCADeformationModelRegistrationMaskCog ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkRigidAndPCADeformationModelRegistrationMaskCog RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkRigidAndPCADeformationModelRegistrationMaskCogNewOpt itkConstrainedRegStepOptimizerForSDM.cxx niftkRigidAndPCADeformationModelRegistrationMaskCogNewOpt.cxx )
TARGET_LINK_LIBRARIES(niftkRigidAndPCADeformationModelRegistrationMaskCogNewOpt ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkRigidAndPCADeformationModelRegistrationMaskCogNewOpt RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkRigidAndPCADeformationModelTransformation niftkRigidAndPCADeformationModelTransformation.cxx )
TARGET_LINK_LIBRARIES(niftkRigidAndPCADeformationModelTransformation ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkRigidAndPCADeformationModelTransformation RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkRigidAndPCADeformationModelTransformationCog niftkRigidAndPCADeformationModelTransformationCog.cxx )
TARGET_LINK_LIBRARIES(niftkRigidAndPCADeformationModelTransformationCog ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkRigidAndPCADeformationModelTransformationCog RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkRigidAndPCADeformationModelTransformationCog3Dto2D niftkRigidAndPCADeformationModelTransformationCog3Dto2D.cxx )
TARGET_LINK_LIBRARIES(niftkRigidAndPCADeformationModelTransformationCog3Dto2D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkRigidAndPCADeformationModelTransformationCog3Dto2D RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkRigidAndPCADeformationModelTransformation3Dto2D niftkRigidAndPCADeformationModelTransformation3Dto2D.cxx )
TARGET_LINK_LIBRARIES(niftkRigidAndPCADeformationModelTransformation3Dto2D ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkRigidAndPCADeformationModelTransformation3Dto2D RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkPCADeformationModelEstimatorDB niftkPCADeformationModelEstimatorDB.cxx )
TARGET_LINK_LIBRARIES(niftkPCADeformationModelEstimatorDB ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkPCADeformationModelEstimatorDB RUNTIME DESTINATION bin COMPONENT applications)

#ADD_EXECUTABLE(niftkSubsampleVectorImage niftkSubsampleVectorImage.cxx )
#TARGET_LINK_LIBRARIES(niftkSubsampleVectorImage ${NIFTK_APPLICATION_LINK_LIBRARIES} )
#INSTALL(TARGETS niftkSubsampleVectorImage RUNTIME DESTINATION bin COMPONENT applications)

################################################################################################
# Command line applications that require VTK 
################################################################################################

IF(VTK_FOUND)

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
      ${Log4cplus_LIB}
      ${Boost_LIBRARIES}
    )

    ADD_EXECUTABLE(niftkCreateBreastMeshMod niftkCreateBreastMeshMod.cxx )
    TARGET_LINK_LIBRARIES(niftkCreateBreastMeshMod ${NIFTK_APPLICATION_WITH_VTK_LINK_LIBRARIES} )
    INSTALL(TARGETS niftkCreateBreastMeshMod RUNTIME DESTINATION bin COMPONENT applications)

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
    ${Log4cplus_LIB}
    ${Boost_LIBRARIES} 
  )

  ###########################################################################
  # Programs that link to NiftySim
  ###########################################################################

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
    ${Log4cplus_LIB}
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

  ADD_EXECUTABLE(niftkNiftySimRegistration3D3D niftkNiftySimRegistration3D3D.cxx )
  TARGET_LINK_LIBRARIES(niftkNiftySimRegistration3D3D ${NIFTK_APPLICATION_WITH_NIFTYSIM_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkNiftySimRegistration3D3D RUNTIME DESTINATION bin COMPONENT applications)

  ADD_EXECUTABLE(niftkNiftySimPlateCompressionRegn2D3D itkConstrainedRegStepOptimizer.cxx niftkNiftySimPlateCompressionRegn2D3D.cxx )
  TARGET_LINK_LIBRARIES(niftkNiftySimPlateCompressionRegn2D3D ${NIFTK_APPLICATION_WITH_NIFTYSIM_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkNiftySimPlateCompressionRegn2D3D RUNTIME DESTINATION bin COMPONENT applications)

  ADD_EXECUTABLE(niftkNiftySimPlateCompressionTransfFrom2Dto3D itkConstrainedRegStepOptimizer.cxx niftkNiftySimPlateCompressionTransfFrom2Dto3D.cxx )
  TARGET_LINK_LIBRARIES(niftkNiftySimPlateCompressionTransfFrom2Dto3D ${NIFTK_APPLICATION_WITH_NIFTYSIM_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkNiftySimPlateCompressionTransfFrom2Dto3D RUNTIME DESTINATION bin COMPONENT applications)

  ADD_EXECUTABLE(niftkNiftySimPlateCompressionTransf2D3D itkConstrainedRegStepOptimizer.cxx niftkNiftySimPlateCompressionTransf2D3D.cxx )
  TARGET_LINK_LIBRARIES(niftkNiftySimPlateCompressionTransf2D3D ${NIFTK_APPLICATION_WITH_NIFTYSIM_LINK_LIBRARIES} )
  INSTALL(TARGETS niftkNiftySimPlateCompressionTransf2D3D RUNTIME DESTINATION bin COMPONENT applications)

ENDIF(CUDA_FOUND)

