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
    NAME tledSolverGPU.h
    PATHS ${NIFTYSIM_DIR}/include
    NO_DEFAULT_PATH
  )
  
  FIND_LIBRARY(NIFTYSIM_LIBRARIES
    NAMES tled
    PATHS ${NIFTYSIM_DIR}/lib
    NO_DEFAULT_PATH
  )

ELSE(CUDA_FOUND)

  FIND_PATH(NIFTYSIM_INCLUDE_DIR
    NAME tledSolverCPU.h
    PATHS ${NIFTYSIM_DIR}/include
    NO_DEFAULT_PATH
  )

  FIND_LIBRARY(NIFTYSIM_LIBRARIES
    NAMES tled
    PATHS ${NIFTYSIM_DIR}/lib
    NO_DEFAULT_PATH
  )

ENDIF(CUDA_FOUND)

IF(NIFTYSIM_LIBRARIES AND NIFTYSIM_INCLUDE_DIR)
  SET(NIFTYSIM_FOUND 1)
ENDIF(NIFTYSIM_LIBRARIES AND NIFTYSIM_INCLUDE_DIR)

