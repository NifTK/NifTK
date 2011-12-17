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

SET(NIFTYREC_FOUND 0)

IF(NOT NIFTYREC_DIR)
  SET(NIFTYREC_DIR ${NIFTK_LINK_PREFIX}/nifty_rec CACHE PATH "Directory containing NiftyRec installation")
ELSE(NOT NIFTYREC_DIR)
  SET(NIFTYREC_DIR @NIFTYREC_DIR@ CACHE PATH "Directory containing NiftyRec installation")
ENDIF(NOT NIFTYREC_DIR)

IF(CUDA_FOUND)

FIND_PATH(NIFTYREC_INCLUDE_DIR
  _et_line_integral_gpu.h
  ${NIFTYREC_DIR}/et-lib_gpu
  ${NIFTYREC_DIR}/et-lib_gpu
  ${NIFTYREC_DIR}/include
  ${NIFTYREC_DIR}-1.1/include
  ${NIFTYREC_DIR}-1.0/include
  @NIFTK_LINK_PREFIX@/include
  /usr/local/include
  /usr/include
  )

FIND_LIBRARY(NIFTYREC_LIBRARIES
  _et_line_integral_gpu
  ${NIFTYREC_DIR}/lib
  ${NIFTYREC_DIR}-1.1/lib
  ${NIFTYREC_DIR}-1.0/lib
  @NIFTK_LINK_PREFIX@/lib
  /usr/local/lib
  /usr/lib
  )

ELSE(CUDA_FOUND)

FIND_PATH(NIFTYREC_INCLUDE_DIR
  _et_line_integral.h
  ${NIFTYREC_DIR}/et-lib
  ${NIFTYREC_DIR}/include
  ${NIFTYREC_DIR}-1.1/include
  ${NIFTYREC_DIR}-1.0/include
  @NIFTK_LINK_PREFIX@/include
  /usr/local/include
  /usr/include
  )

FIND_LIBRARY(NIFTYREC_LIBRARIES
  _et_line_integral
  ${NIFTYREC_DIR}/lib
  ${NIFTYREC_DIR}-1.1/lib
  ${NIFTYREC_DIR}-1.0/lib
  @NIFTK_LINK_PREFIX@/lib
  /usr/local/lib
  /usr/lib
  )

ENDIF(CUDA_FOUND)

IF(NIFTYREC_LIBRARIES AND NIFTYREC_INCLUDE_DIR)
  SET(NIFTYREC_FOUND 1)
ENDIF(NIFTYREC_LIBRARIES AND NIFTYREC_INCLUDE_DIR)
