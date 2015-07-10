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

project(uk_ac_ucl_cmic_breastsegmentation)

find_package(NiftySeg)

if(NiftySeg_FOUND)

  include_directories(${NiftySeg_INCLUDE_DIR})
  add_definitions(-DUSE_NIFTYSEG)
  set(NIFTYSEG_NVCC_FLAGS "${NIFTYSEG_NVCC_FLAGS};-DUSE_NIFTYSEG")

  mitk_create_plugin(
    EXPORT_DIRECTIVE BREASTSEGMENTATION_EXPORT
    EXPORTED_INCLUDE_SUFFIXES src
    MODULE_DEPENDS MitkQtWidgetsExt niftkCore niftkCoreGui
    )

  target_link_libraries(${PROJECT_NAME} PRIVATE niftkITK ${NiftySeg_LIBRARIES})

  if (NIFTK_USE_COTIRE AND COMMAND cotire)
    cotire(uk_ac_ucl_cmic_breastsegmentation)
  endif()

else()

  message("Didn't find NiftySeg so cannot build breast segmentation plugin")

endif()

