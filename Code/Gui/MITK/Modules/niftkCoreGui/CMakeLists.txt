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

set(_modules MitkCore MitkQtWidgets MitkQtWidgetsExt niftkCore)
set(_packages Qt4|QtGui Qt4|QtOpenGL CTK|CTKWidgets)

#if(BUILD_PCL)
#  # pcl depends on eigen
#  list(APPEND _packages Eigen)
#  list(APPEND _packages PCL)
#endif()
#
#if(BUILD_VL)
#  find_package(VL COMPONENTS VLCore VLGraphics VLVolume VLQt4 REQUIRED)
#  link_directories(${VL_LIBRARIES})
#  include_directories(${VL_INCLUDE_DIRS})
#
#  # render code needs opencl for translucent triangle sorting.
#  set(_modules
#    ${_modules}
#    MitkOpenCL
#  )
#
#  # vl renderer can use calibrated cameras
#  if(BUILD_IGI)
#    set(_modules ${_modules} niftkOpenCV MitkCameraCalibration)
#  endif()
#endif()
#
#if(CUDA_FOUND AND NIFTK_USE_CUDA)
#  # if cuda is available then it can dump the renderer output into datastorage.
#  set(_modules
#    ${_modules}
#    niftkCUDA
#  )
#endif()

MITK_CREATE_MODULE(niftkCoreGui
  EXPORT_DEFINE NIFTKCOREGUI_EXPORT
  INCLUDE_DIRS Events LookupTables Rendering
  DEPENDS ${_modules}
  PACKAGE_DEPENDS ${_packages}
)

target_link_libraries(niftkCoreGui
  niftkITK
  niftkVTK
)

#if(BUILD_VL)
#  target_link_libraries(niftkCoreGui
#    ${VL_LIBRARIES}
#  )
#endif()


if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkCoreGui)
endif()

if(BUILD_TESTING)
  #add_subdirectory(Testing)
endif(BUILD_TESTING)
