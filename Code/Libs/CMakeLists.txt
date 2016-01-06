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

if(CUDA_FOUND AND NIFTK_USE_CUDA)
  add_subdirectory(CUDAKernels)
endif()

add_subdirectory(Common)

find_package(ITK)
if(ITK_FOUND)
  include(${MITK_MODULES_PACKAGE_DEPENDS_DIR}/MITK_ITK_Config.cmake)
  add_subdirectory(ITK)
  add_subdirectory(ITKIO)
endif(ITK_FOUND)

find_package(VTK REQUIRED)
if(VTK_FOUND)
  include(${VTK_USE_FILE})
  add_subdirectory(VTK)
endif(VTK_FOUND)

if(NVAPI_FOUND AND CUDA_FOUND AND NIFTK_USE_CUDA)
  add_subdirectory(NVAPI)
endif()

# Meshing requires ITK, VTK: DO NOT MOVE BEFORE FIND_PACKAGE(ITK), ...
if (BUILD_MESHING)
  add_subdirectory(Meshing)
endif (BUILD_MESHING)
