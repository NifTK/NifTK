#/*================================================================================
#
#  UCLToolkit: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details.
#
#  Last Changed      : $LastChangedDate: 2010-11-10 15:46:11 +0000 (Mi, 10 Nov 2010) $
#  Revision          : $Revision: 4163 $
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


# Find ITK.
FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

SET( NIFTK_APPLICATION_LINK_LIBRARIES
      niftkcommon
      niftkITK
      ITKCommon
      ITKIO
      ITKBasicFilters
      ITKAlgorithms
      itkzlib
      ${Boost_LIBRARIES} )
  
SET( NIFTYSIM_LINK_LIBRARIES 
       tled
       xmlParser )


SET( VTK_LINK_LIBRARIES
        vtkIO
        vtkCommon
        vtkGraphics
        vtkImaging )
  


          
#
# Composition of 3 images to 1 vector image
#

ADD_EXECUTABLE( niftkCompose3DVectorImageFromComponentImages
                  niftkCompose3DVectorImageFromComponentImages.cxx )

TARGET_LINK_LIBRARIES( niftkCompose3DVectorImageFromComponentImages
                         ${NIFTK_APPLICATION_LINK_LIBRARIES} )

INSTALL( TARGETS niftkCompose3DVectorImageFromComponentImages RUNTIME DESTINATION bin COMPONENT applications)                            
                         
                         
#
# Deformation field TRE evaluation
#

ADD_EXECUTABLE( niftkDeformationFieldTargetRegistrationErrorWithHistogram
                  niftkDeformationFieldTargetRegistrationErrorWithHistogram.cxx )

TARGET_LINK_LIBRARIES( niftkDeformationFieldTargetRegistrationErrorWithHistogram
                         ${NIFTK_APPLICATION_LINK_LIBRARIES} )
                         
INSTALL( TARGETS niftkDeformationFieldTargetRegistrationErrorWithHistogram RUNTIME DESTINATION bin COMPONENT applications )                            

#
# Test of image handling
#

ADD_EXECUTABLE( niftkTestNiiGeneration
                  niftkTestNiiGeneration.cxx )

TARGET_LINK_LIBRARIES( niftkTestNiiGeneration
                         ${NIFTK_APPLICATION_LINK_LIBRARIES} )
                         
#INSTALL( TARGETS niftkTestNiiGeneration RUNTIME DESTINATION bin COMPONENT applications)                                                  


#
# Conversion forcing vector image type within the reader...
#

ADD_LIBRARY( bePrototypeUtils STATIC
               ITKppDofFileReader.cxx
               ITKppImage2WorldMatrixCalculator )

#ADD_EXECUTABLE( niftkConvertNiftiVectorImage
#                  niftkConvertNiftiVectorImage.cxx )

#TARGET_LINK_LIBRARIES( niftkConvertNiftiVectorImage
#                         bePrototypeUtils
#                         ${NIFTK_APPLICATION_LINK_LIBRARIES} )
                         
#INSTALL( TARGETS niftkConvertNiftiVectorImage RUNTIME DESTINATION bin COMPONENT applications )                         


#
# Validation of the vector field deformation
#

ADD_EXECUTABLE( niftkValidateDeformationVectorField
                  niftkValidateDeformationVectorField.cxx )

TARGET_LINK_LIBRARIES( niftkValidateDeformationVectorField
                         ${NIFTK_APPLICATION_LINK_LIBRARIES} )

INSTALL( TARGETS niftkValidateDeformationVectorField RUNTIME DESTINATION bin COMPONENT applications )

#
# Generation of the homogenous matrix from itk++ dof file (rigid 3D)
#

ADD_EXECUTABLE( niftkITKppRigidDofConversion
                  niftkITKppRigidDofConversion.cxx )

TARGET_LINK_LIBRARIES( niftkITKppRigidDofConversion
                         bePrototypeUtils
                         ${NIFTK_APPLICATION_LINK_LIBRARIES} )

INSTALL( TARGETS niftkITKppRigidDofConversion RUNTIME DESTINATION bin COMPONENT applications )


#
# deform an image according to a simulation run by niftySim
# this bit requires niftySim and possibly cuda...
#
# IF( NIFTK_USE_NIFTYSIM )
    # IF( CUDA_FOUND ) 
        # MESSAGE( BE-Prototype CUDA_NVCC_INCLUDE_ARGS_USER: ${CUDA_NVCC_INCLUDE_ARGS_USER} )
        # MESSAGE( BE-Prototype CUDA_NVCC_INCLUDE_ARGS: ${CUDA_NVCC_INCLUDE_ARGS} ) 
        # MESSAGE( BE-Prototype CUDA_TOOLKIT_INCLUDE: ${CUDA_TOOLKIT_INCLUDE} )
        # CUDA_ADD_EXECUTABLE( niftkDeformImageFromNiftySimulation 
                               # niftkDeformImageFromNiftySimulation.cxx )

        # TARGET_LINK_LIBRARIES( niftkDeformImageFromNiftySimulation 
                                 # bePrototypeUtils
                                 # ${NIFTK_APPLICATION_LINK_LIBRARIES}
                                 # ${NIFTYSIM_LIBRARIES} )
        # INSTALL( TARGETS niftkDeformImageFromNiftySimulation RUNTIME DESTINATION bin COMPONENT applications )
        
    # ELSE( CUDA_FOUND )
            # ADD_EXECUTABLE( niftkDeformImageFromNiftySimulation 
                              # niftkDeformImageFromNiftySimulation.cxx )
            # TARGET_LINK_LIBRARIES( niftkDeformImageFromNiftySimulation 
                             # bePrototypeUtils
                             # ${NIFTK_APPLICATION_LINK_LIBRARIES}
                             # ${NIFTYSIM_LIBRARIES} )
            # INSTALL( TARGETS niftkDeformImageFromNiftySimulation RUNTIME DESTINATION bin COMPONENT applications )
                             
    # ENDIF( CUDA_FOUND )
# ENDIF( NIFTK_USE_NIFTYSIM )

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
  LINK_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/C/common/lib/x64 ${CUDA_SDK_ROOT_DIR}/shared/lib/x64)
  
  SET(NIFTK_APPLICATION_WITH_CUDA_LINK_LIBRARIES
    niftkcommon
    niftkITK
    niftkcudautilsgpu 
    ${Boost_LIBRARIES} 
  )

  #CUDA_ADD_EXECUTABLE( niftkCUDAGLVolumeRender niftkCUDAGLVolumeRender.cxx )
  #TARGET_LINK_LIBRARIES( niftkCUDAGLVolumeRender ${NIFTK_APPLICATION_WITH_CUDA_LINK_LIBRARIES} )
  #INSTALL(TARGETS niftkCUDAGLVolumeRender RUNTIME DESTINATION bin COMPONENT applications)


  ###########################################################################
  # Programs that link to NiftySim
  ###########################################################################

  IF(NIFTYSIM_FOUND)

    INCLUDE(${CMAKE_SOURCE_DIR}/CMake/FindNiftySim.cmake)

    INCLUDE_DIRECTORIES(${NIFTYSIM_INCLUDE_DIR})
    LINK_DIRECTORIES(${NIFTYSIM_DIR}/lib /opt/fontconfig/lib/)

    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D_GPU_")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GPU_")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
  
    SET(NIFTK_APPLICATION_WITH_NIFTYSIM_LINK_LIBRARIES
      ${NIFTK_APPLICATION_WITH_CUDA_LINK_LIBRARIES}
      tled
      xmlParser
      )
  
    IF(VTK_FOUND)
      #IF(QT_FOUND)

        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_Visualisation_")
          
        SET(NIFTK_APPLICATION_WITH_NIFTYSIM_LINK_LIBRARIES 
          ${NIFTK_APPLICATION_WITH_NIFTYSIM_LINK_LIBRARIES}
          viz
          #niftkITKVTK
          niftkVTK
          vtkHybrid
          vtkRendering
          vtkGraphics
          vtkImaging
          vtkIO
          vtkCommon
          vtkVolumeRendering
          #QVTK
          #${QT_LIBRARIES} 
          )
      
      #ENDIF(QT_FOUND)
    ENDIF(VTK_FOUND)


    ADD_EXECUTABLE(niftkDeformImageFromNiftySimulation niftkDeformImageFromNiftySimulation.cxx )
    TARGET_LINK_LIBRARIES(niftkDeformImageFromNiftySimulation ${NIFTK_APPLICATION_WITH_NIFTYSIM_LINK_LIBRARIES} )
    INSTALL(TARGETS niftkDeformImageFromNiftySimulation RUNTIME DESTINATION bin COMPONENT applications)

  ENDIF( NIFTYSIM_FOUND )

ENDIF(CUDA_FOUND)


