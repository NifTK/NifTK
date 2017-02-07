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

if(NVAPI_FOUND AND CUDA_FOUND AND NIFTK_USE_CUDA)

  if(NOT (CUDA_VERSION_MAJOR LESS 7))
    include_directories(${CMAKE_SOURCE_DIR}/Libraries/NVAPI/nv7)
    ADD_DEFINITIONS(-DUSE_NV_7)
    ADD_DEFINITIONS(-DGLEW_STATIC)
  endif(NOT (CUDA_VERSION_MAJOR LESS 7)) 

  MITK_CREATE_MODULE(
    DEPENDS niftkIGIDataSources niftkIGIGui
    PACKAGE_DEPENDS Qt4|QtCore Qt5|Widgets
    AUTOLOAD_WITH niftkIGIDataSources
  )  
  
  include_directories(${CMAKE_SOURCE_DIR}/Libraries/NVAPI/video)    
  include_directories(${CMAKE_SOURCE_DIR}/Libraries/NVAPI/glew)		  
  include_directories(${QT_QTOPENGL_INCLUDE_DIR})
  include_directories(${CUDA_TOOLKIT_INCLUDE})

  # stuffing this into DEPENDS above doesnt work for some reason
  target_link_libraries(niftkNVidiaSDIDataSourceService PRIVATE
    niftknvapi
    Winmm.lib
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
  )

  if(MSVC AND NIFTK_DELAYLOAD_CUDA)
  # all of the nvidia-driver/cuda dependencies should be delay-loaded
  #  so this library would at least load on a machine that doesnt have nv hardware
  set_target_properties(niftkNVidiaSDIDataSourceService PROPERTIES LINK_FLAGS
    "/DELAYLOAD:nvcuda.dll /DELAYLOAD:${CUDA_CUDART_DLL_NAME}")
  endif()

  if (NIFTK_USE_COTIRE AND COMMAND cotire)
    cotire(niftkNVidiaSDIDataSourceService)
  endif()

endif()

