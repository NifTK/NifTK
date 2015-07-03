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

# render code needs opencl for translucent triangle sorting.
set(_modules MitkOpenCL niftkCoreGui)
set(_packages Qt4|QtGui Qt4|QtOpenGL)

if(BUILD_PCL)
  # pcl depends on eigen
  list(APPEND _packages Eigen)
  list(APPEND _packages PCL)
endif()


find_package(VL COMPONENTS VLCore VLGraphics VLVolume VLQt4 REQUIRED)
link_directories(${VL_LIBRARIES})
include_directories(${VL_INCLUDE_DIRS})

# vl renderer can use calibrated cameras
if(BUILD_IGI)
  set(_modules ${_modules} niftkOpenCV MitkCameraCalibration)
endif()

# vl renderer can draw nice textured point clouds.
if(BUILD_PCL)
  set(_modules ${_modules} niftkPCL)
endif()

if(CUDA_FOUND AND NIFTK_USE_CUDA)
  # if cuda is available then it can dump the renderer output into datastorage.
  set(_modules
    ${_modules}
    niftkCUDA
  )
endif()

MITK_CREATE_MODULE(niftkVL
  INCLUDE_DIRS Rendering
  DEPENDS ${_modules}
  PACKAGE_DEPENDS ${_packages}
)

target_link_libraries(niftkVL PRIVATE ${VL_LIBRARIES})

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkVL)
endif()

if(BUILD_TESTING)
  #add_subdirectory(Testing)
endif(BUILD_TESTING)
