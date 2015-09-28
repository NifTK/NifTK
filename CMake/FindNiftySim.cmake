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


if (NOT NiftySim_FOUND)

  set(NiftySim_DIR @NiftySim_DIR@ CACHE PATH "Directory containing NiftySim installation")

  if(CUDA_FOUND AND NiftySim_USE_CUDA)

    find_path(NiftySim_INCLUDE_DIR
      NAME tledSolverGPU.h
      PATHS ${NiftySim_DIR}/include
      NO_DEFAULT_PATH
    )

    find_library(NiftySim_LIBRARIES
      NAMES tled tled${NiftySim_DEBUG_POSTFIX}
      PATHS ${NiftySim_DIR}/lib
      NO_DEFAULT_PATH
    )

  else()

    find_path(NiftySim_INCLUDE_DIR
      NAME tledSolverCPU.h
      PATHS ${NiftySim_DIR}/include
      NO_DEFAULT_PATH
    )

    find_library(NiftySim_LIBRARIES
      NAMES tled tled${NiftySim_DEBUG_POSTFIX}
      PATHS ${NiftySim_DIR}/lib
      NO_DEFAULT_PATH
    )

  endif()

  if(NiftySim_LIBRARIES AND NiftySim_INCLUDE_DIR)
    set(NiftySim_FOUND 1)
  endif(NiftySim_LIBRARIES AND NiftySim_INCLUDE_DIR)

endif()
