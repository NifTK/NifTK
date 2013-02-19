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


SET(NIFTYSIM_FOUND 0)

IF(NOT NIFTYSIM_DIR)
  SET(NIFTYSIM_DIR ${NIFTK_LINK_PREFIX}/nifty_sim CACHE PATH "Directory containing NiftySim installation")
ELSE(NOT NIFTYSIM_DIR)
  SET(NIFTYSIM_DIR @NIFTYSIM_DIR@ CACHE PATH "Directory containing NiftySim installation")
ENDIF(NOT NIFTYSIM_DIR)

IF(CUDA_FOUND)

  FIND_PATH(NIFTYSIM_INCLUDE_DIR
    tledSolverGPU.h
    ${NIFTYSIM_DIR}/include
    ${NIFTYSIM_DIR}-1.0/include
    @NIFTK_LINK_PREFIX@/include
    /usr/local/include
    /usr/include
  )
  
  FIND_LIBRARY(NIFTYSIM_LIBRARIES
    tled
    ${NIFTYSIM_DIR}/lib
    NIFTYSIM_DIR  ${NIFTYSIM_DIR}-1.0/lib
    @NIFTK_LINK_PREFIX@/lib
    /usr/local/lib
    /usr/lib
  )

ELSE(CUDA_FOUND)

  FIND_PATH(NIFTYSIM_INCLUDE_DIR
    tledSolverCPU.h
    ${NIFTYSIM_DIR}/include
    ${NIFTYSIM_DIR}-1.0/include
    @NIFTK_LINK_PREFIX@/include
    /usr/local/include
    /usr/include
  )

  FIND_LIBRARY(NIFTYSIM_LIBRARIES
    tled
    ${NIFTYSIM_DIR}/lib
    ${NIFTYSIM_DIR}-1.0/lib
    @NIFTK_LINK_PREFIX@/lib
    /usr/local/lib
    /usr/lib
  )

ENDIF(CUDA_FOUND)

IF(NIFTYSIM_LIBRARIES AND NIFTYSIM_INCLUDE_DIR)
  SET(NIFTYSIM_FOUND 1)
ENDIF(NIFTYSIM_LIBRARIES AND NIFTYSIM_INCLUDE_DIR)

