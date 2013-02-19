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
    _seg_tools.h
    ${NIFTYSEG_DIR}/include
    ${NIFTYSEG_DIR}-1.0/include
    @NIFTK_LINK_PREFIX@/include
    /usr/local/include
    /usr/include
    )
  
  FIND_LIBRARY(NIFTYSEG_LIBRARIES
    _seg_nifti
    ${NIFTYSEG_DIR}/lib
    ${NIFTYSEG_DIR}-1.0/lib
    @NIFTK_LINK_PREFIX@/lib
    /usr/local/lib
    /usr/lib
    )

ELSE(CUDA_FOUND)

  FIND_PATH(NIFTYSEG_INCLUDE_DIR
    _seg_tools.h
    ${NIFTYSEG_DIR}/include
    ${NIFTYSEG_DIR}-1.0/include
    @NIFTK_LINK_PREFIX@/include
    /usr/local/include
    /usr/include
    )
  
  FIND_LIBRARY(NIFTYSEG_LIBRARIES
    _seg_nifti
    ${NIFTYSEG_DIR}/lib
    ${NIFTYSEG_DIR}-1.0/lib
    @NIFTK_LINK_PREFIX@/lib
    /usr/local/lib
    /usr/lib
    )

ENDIF(CUDA_FOUND)

IF(NIFTYSEG_LIBRARIES AND NIFTYSEG_INCLUDE_DIR)
  SET(NIFTYSEG_FOUND 1)
ENDIF(NIFTYSEG_LIBRARIES AND NIFTYSEG_INCLUDE_DIR)
