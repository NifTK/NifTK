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

find_package(CUDA REQUIRED)

MITK_CREATE_MODULE(
  DEPENDS niftkIGIDataSources niftkIGIGui
)

include_directories(${CMAKE_SOURCE_DIR}/Code/Libs/NVAPI/video)
include_directories(${CMAKE_SOURCE_DIR}/Code/Libs/NVAPI/glew)
include_directories(${QT_QTOPENGL_INCLUDE_DIR})
include_directories(${CUDA_TOOLKIT_INCLUDE})

# stuffing this into DEPENDS above doesnt work for some reason
target_link_libraries(niftkNVidia PRIVATE
  niftknvapi
  MitkOpenCVVideoSupport
  Winmm.lib
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  )

if(MSVC AND NIFTK_DELAYLOAD_CUDA)
  # all of the nvidia-driver/cuda dependencies should be delay-loaded
  #  so this library would at least load on a machine that doesnt have nv hardware
  set_target_properties(niftkNVidia PROPERTIES LINK_FLAGS 
    "/DELAYLOAD:nvcuda.dll /DELAYLOAD:${CUDA_CUDART_DLL_NAME}")
endif()

if (NIFTK_USE_COTIRE AND COMMAND cotire)
 # cotire(niftkNVidia)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
