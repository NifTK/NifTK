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


SET(NIFTYREC_FOUND 0)

IF(NOT NIFTYREC_DIR)
  SET(NIFTYREC_DIR ${NIFTK_LINK_PREFIX}/nifty_rec CACHE PATH "Directory containing NiftyRec installation")
ELSE(NOT NIFTYREC_DIR)
  SET(NIFTYREC_DIR @NIFTYREC_DIR@ CACHE PATH "Directory containing NiftyRec installation")
ENDIF(NOT NIFTYREC_DIR)

IF(CUDA_FOUND)

FIND_PATH(NIFTYREC_INCLUDE_DIR
  NAME _et_line_integral_gpu.h
  PATHS ${NIFTYREC_DIR}/et-lib_gpu
        ${NIFTYREC_DIR}/include
  NO_DEFAULT_PATH
  )

FIND_LIBRARY(NIFTYREC_LIBRARIES
  NAMES _et_line_integral_gpu
  PATHS ${NIFTYREC_DIR}/lib
  NO_DEFAULT_PATH
  )

ELSE(CUDA_FOUND)

FIND_PATH(NIFTYREC_INCLUDE_DIR
  NAME _et_line_integral.h
  PATHS ${NIFTYREC_DIR}/et-lib
        ${NIFTYREC_DIR}/include
  NO_DEFAULT_PATH
  )

FIND_LIBRARY(NIFTYREC_LIBRARIES
  NAMES _et_line_integral
  PATHS ${NIFTYREC_DIR}/lib
  NO_DEFAULT_PATH
  )

ENDIF(CUDA_FOUND)

IF(NIFTYREC_LIBRARIES AND NIFTYREC_INCLUDE_DIR)
  SET(NIFTYREC_FOUND 1)
ENDIF(NIFTYREC_LIBRARIES AND NIFTYREC_INCLUDE_DIR)
