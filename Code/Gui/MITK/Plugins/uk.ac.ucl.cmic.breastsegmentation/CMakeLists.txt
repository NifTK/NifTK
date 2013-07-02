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

if(NIFTYSEG_FOUND)

  include_directories(${NIFTYSEG_INCLUDE_DIR})
  add_definitions(-DUSE_NIFTYSEG)
  set(NIFTYSEG_NVCC_FLAGS "${NIFTYSEG_NVCC_FLAGS};-DUSE_NIFTYSEG")

  MACRO_CREATE_MITK_CTK_PLUGIN(
    EXPORT_DIRECTIVE BREASTSEGMENTATION_EXPORT
    EXPORTED_INCLUDE_SUFFIXES src
    MODULE_DEPENDENCIES CTK QmitkExt niftkCore niftkCoreGui
    )

  target_link_libraries(${PROJECT_NAME}
    niftkITK    
    ${NIFTYSEG_LIBRARIES}
    )

else(NIFTYSEG_FOUND)

  message("Didn't find NiftySeg so cannot build breast segmentation plugin")

endif(NIFTYSEG_FOUND)

