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


set(NIFTYREC_FOUND 0)

if(NOT NIFTYREC_DIR)
  set(NIFTYREC_DIR ${NIFTK_LINK_PREFIX}/nifty_rec CACHE PATH "Directory containing NiftyRec installation")
else(NOT NIFTYREC_DIR)
  set(NIFTYREC_DIR @NIFTYREC_DIR@ CACHE PATH "Directory containing NiftyRec installation")
endif(NOT NIFTYREC_DIR)

if(CUDA_FOUND)

find_path(NIFTYREC_INCLUDE_DIR
  NAME _et_line_integral_gpu.h
  PATHS ${NIFTYREC_DIR}/et-lib_gpu
        ${NIFTYREC_DIR}/include
  NO_DEFAULT_PATH
  )

find_library(NIFTYREC_LIBRARIES
  NAMES _et_line_integral_gpu
  PATHS ${NIFTYREC_DIR}/lib
  NO_DEFAULT_PATH
  )

else(CUDA_FOUND)

find_path(NIFTYREC_INCLUDE_DIR
  NAME _et_line_integral.h
  PATHS ${NIFTYREC_DIR}/et-lib
        ${NIFTYREC_DIR}/include
  NO_DEFAULT_PATH
  )

find_library(NIFTYREC_LIBRARIES
  NAMES _et_line_integral
  PATHS ${NIFTYREC_DIR}/lib
  NO_DEFAULT_PATH
  )

endif(CUDA_FOUND)

if(NIFTYREC_LIBRARIES AND NIFTYREC_INCLUDE_DIR)
  set(NIFTYREC_FOUND 1)
endif(NIFTYREC_LIBRARIES AND NIFTYREC_INCLUDE_DIR)
