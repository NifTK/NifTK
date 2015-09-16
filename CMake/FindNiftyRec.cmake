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


if (NOT NiftyRec_FOUND)

  set(NiftyRec_DIR @NiftyRec_DIR@ CACHE PATH "Directory containing NiftyRec installation")

  if(CUDA_FOUND)

    find_path(NiftyRec_INCLUDE_DIR
      NAME _et_line_integral_gpu.h
      PATHS ${NiftyRec_DIR}/et-lib_gpu
            ${NiftyRec_DIR}/include
      NO_DEFAULT_PATH
    )

    find_library(NiftyRec_LIBRARIES
      NAMES _et_line_integral_gpu _et_line_integral_gpu${NiftyRec_DEBUG_POSTFIX}
      PATHS ${NiftyRec_DIR}/lib
      NO_DEFAULT_PATH
    )

  else(CUDA_FOUND)

    find_path(NiftyRec_INCLUDE_DIR
      NAME _et_line_integral.h
      PATHS ${NiftyRec_DIR}/et-lib
            ${NiftyRec_DIR}/include
      NO_DEFAULT_PATH
    )

    find_library(NiftyRec_LIBRARIES
      NAMES _et_line_integral _et_line_integral${NiftyRec_DEBUG_POSTFIX}
      PATHS ${NiftyRec_DIR}/lib
      NO_DEFAULT_PATH
    )

  endif(CUDA_FOUND)

  if(NiftyRec_LIBRARIES AND NiftyRec_INCLUDE_DIR)
    set(NiftyRec_FOUND 1)
  endif()

endif()
