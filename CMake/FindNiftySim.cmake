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


set(NIFTYSIM_FOUND 0)

if(NOT NIFTYSIM_DIR)
  set(NIFTYSIM_DIR ${NIFTK_LINK_PREFIX}/nifty_sim CACHE PATH "Directory containing NiftySim installation")
else(NOT NIFTYSIM_DIR)
  set(NIFTYSIM_DIR @NIFTYSIM_DIR@ CACHE PATH "Directory containing NiftySim installation")
endif(NOT NIFTYSIM_DIR)

if(CUDA_FOUND)

  find_path(NIFTYSIM_INCLUDE_DIR
    NAME tledSolverGPU.h
    PATHS ${NIFTYSIM_DIR}/include
    NO_DEFAULT_PATH
  )
  
  find_library(NIFTYSIM_LIBRARIES
    NAMES tled
    PATHS ${NIFTYSIM_DIR}/lib
    NO_DEFAULT_PATH
  )

else(CUDA_FOUND)

  find_path(NIFTYSIM_INCLUDE_DIR
    NAME tledSolverCPU.h
    PATHS ${NIFTYSIM_DIR}/include
    NO_DEFAULT_PATH
  )

  find_library(NIFTYSIM_LIBRARIES
    NAMES tled
    PATHS ${NIFTYSIM_DIR}/lib
    NO_DEFAULT_PATH
  )

endif(CUDA_FOUND)

if(NIFTYSIM_LIBRARIES AND NIFTYSIM_INCLUDE_DIR)
  set(NIFTYSIM_FOUND 1)
endif(NIFTYSIM_LIBRARIES AND NIFTYSIM_INCLUDE_DIR)

