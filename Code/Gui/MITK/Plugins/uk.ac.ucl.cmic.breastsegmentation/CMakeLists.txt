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

FIND_PACKAGE(NiftySeg)

IF(NIFTYSEG_FOUND)

  INCLUDE_DIRECTORIES(${NIFTYSEG_INCLUDE_DIR})
  ADD_DEFINITIONS(-DUSE_NIFTYSEG)
  SET(NIFTYSEG_NVCC_FLAGS "${NIFTYSEG_NVCC_FLAGS};-DUSE_NIFTYSEG")

  MACRO_CREATE_MITK_CTK_PLUGIN(
    EXPORT_DIRECTIVE BREASTSEGMENTATION_EXPORT
    EXPORTED_INCLUDE_SUFFIXES src
    MODULE_DEPENDENCIES CTK QmitkExt niftkCore niftkCoreGui
    )

  TARGET_LINK_LIBRARIES(${PROJECT_NAME}
    niftkITK    
    ${NIFTYSEG_LIBRARIES}
    )

ELSE(NIFTYSEG_FOUND)

  MESSAGE("Didn't find NiftySeg so cannot build breast segmentation plugin")

ENDIF(NIFTYSEG_FOUND)

