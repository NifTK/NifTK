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


SET(NIFTYSEG_FOUND 0)

IF(NOT NIFTYSEG_DIR)
  SET(NIFTYSEG_DIR ${NIFTK_LINK_PREFIX}/nifty_seg CACHE PATH "Directory containing NiftySeg installation")
ELSE(NOT NIFTYSEG_DIR)
  SET(NIFTYSEG_DIR @NIFTYSEG_DIR@ CACHE PATH "Directory containing NiftySeg installation")
ENDIF(NOT NIFTYSEG_DIR)

IF(CUDA_FOUND)

  FIND_PATH(NIFTYSEG_INCLUDE_DIR
    NAME _seg_tools.h
    PATHS ${NIFTYSEG_DIR}/include
    NO_DEFAULT_PATH
    )
  
  FIND_LIBRARY(NIFTYSEG_LIBRARIES
    NAMES _seg_nifti
    PATHS ${NIFTYSEG_DIR}/lib
    NO_DEFAULT_PATH
    )

ELSE(CUDA_FOUND)

  FIND_PATH(NIFTYSEG_INCLUDE_DIR
    NAME _seg_tools.h
    PATHS ${NIFTYSEG_DIR}/include
    NO_DEFAULT_PATH
    )
  
  FIND_LIBRARY(NIFTYSEG_LIBRARIES
    NAMES _seg_nifti
    PATHS ${NIFTYSEG_DIR}/lib
    NO_DEFAULT_PATH
    )

ENDIF(CUDA_FOUND)

IF(NIFTYSEG_LIBRARIES AND NIFTYSEG_INCLUDE_DIR)
  SET(NIFTYSEG_FOUND 1)
ENDIF(NIFTYSEG_LIBRARIES AND NIFTYSEG_INCLUDE_DIR)
