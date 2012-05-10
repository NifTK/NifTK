#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#  
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: $
#  Revision          : $Revision: $
#  Last modified by  : $Author: jhh $
#
#  Original author   : j.hipwell@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

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
