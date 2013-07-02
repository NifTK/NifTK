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


set(NIFTYSEG_FOUND 0)

if(NOT NIFTYSEG_DIR)
  set(NIFTYSEG_DIR ${NIFTK_LINK_PREFIX}/nifty_seg CACHE PATH "Directory containing NiftySeg installation")
else(NOT NIFTYSEG_DIR)
  set(NIFTYSEG_DIR @NIFTYSEG_DIR@ CACHE PATH "Directory containing NiftySeg installation")
endif(NOT NIFTYSEG_DIR)

if(CUDA_FOUND)

  find_path(NIFTYSEG_INCLUDE_DIR
    NAME _seg_tools.h
    PATHS ${NIFTYSEG_DIR}/include
    NO_DEFAULT_PATH
    )
  
  find_library(NIFTYSEG_LIBRARIES
    NAMES _seg_nifti
    PATHS ${NIFTYSEG_DIR}/lib
    NO_DEFAULT_PATH
    )

else(CUDA_FOUND)

  find_path(NIFTYSEG_INCLUDE_DIR
    NAME _seg_tools.h
    PATHS ${NIFTYSEG_DIR}/include
    NO_DEFAULT_PATH
    )
  
  find_library(NIFTYSEG_LIBRARIES
    NAMES _seg_nifti
    PATHS ${NIFTYSEG_DIR}/lib
    NO_DEFAULT_PATH
    )

endif(CUDA_FOUND)

if(NIFTYSEG_LIBRARIES AND NIFTYSEG_INCLUDE_DIR)
  set(NIFTYSEG_FOUND 1)
endif(NIFTYSEG_LIBRARIES AND NIFTYSEG_INCLUDE_DIR)
