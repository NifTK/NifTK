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

project(uk_ac_ucl_cmic_vlrenderer)

set(_depends MitkCore MitkQtWidgets MitkQtWidgetsExt niftkCore niftkCoreGui)

if(CUDA_FOUND AND NIFTK_USE_CUDA)
  list(APPEND _depends niftkCUDA)
endif()


# not sure if this is necessary here.
find_package(VL COMPONENTS VLCore VLGraphics VLVolume VLQt4 REQUIRED)
link_directories(${VL_LIBRARIES})
include_directories(${VL_INCLUDE_DIRS})


MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE NEWVISUALIZATION_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDS ${_depends}
)

# not nice...
if(CUDA_FOUND AND NIFTK_USE_CUDA)
  include_directories(${CUDA_TOOLKIT_INCLUDE} ${CMAKE_SOURCE_DIR}/Code/Libs/CUDAKernels)
  target_link_libraries(uk_ac_ucl_cmic_vlrenderer ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} niftkCUDAKernels)
endif()

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(uk_ac_ucl_cmic_vlrenderer)
endif()
