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

project(uk_ac_ucl_cmic_niftyreg)

LINK_DIRECTORIES(${NIFTYREG_LIBRARY_DIR})
INCLUDE_DIRECTORIES(${NIFTYREG_INCLUDE_DIR})
ADD_DEFINITIONS(-DUSE_NIFTYREG)

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE NIFTYREG_EXPORTS
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES CTK QmitkExt niftkCore niftkCoreGui
)

if(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    # OpenMP is disabled when the compiler is based on Clang
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif(OPENMP_FOUND)
endif(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")

TARGET_LINK_LIBRARIES(${PROJECT_NAME} 
                      niftkITK
                      niftkVTK
		      ${NIFTYREG_LIBRARIES}
)

# The NiftyReg plugin currently can't use the CUDA version of NiftyReg
# because of problems compiling the gpu code as shared libraries
if(FALSE AND NIFTK_USE_CUDA)

  FIND_PACKAGE(CUDA)
  IF(CUDA_FOUND)
    ADD_DEFINITIONS(-D_USE_CUDA)
  ENDIF(CUDA_FOUND)
  
  TARGET_LINK_LIBRARIES(${PROJECT_NAME} 
			_reg_aladin
			_reg_f3d
			_reg_thinPlateSpline
			_reg_blockMatching_gpu
                        _reg_resampling_gpu 
			_reg_localTransformation_gpu
                        _reg_globalTransformation_gpu 
                        _reg_mutualinformation_gpu 
                        _reg_tools_gpu 
                        _reg_cudaCommon)
endif(FALSE AND NIFTK_USE_CUDA)


