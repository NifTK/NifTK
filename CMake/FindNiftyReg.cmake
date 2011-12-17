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

SET(NIFTYREG_FOUND 0)

IF(NOT NIFTYREG_DIR)
  SET(NIFTYREG_DIR ${NIFTK_LINK_PREFIX}/nifty_reg CACHE PATH "Directory containing NiftyReg installation")
ELSE(NOT NIFTYREG_DIR)
  SET(NIFTYREG_DIR @NIFTYREG_DIR@ CACHE PATH "Directory containing NiftyReg installation")
ENDIF(NOT NIFTYREG_DIR)

IF(CUDA_FOUND)

  FIND_PATH(NIFTYREG_INCLUDE_DIR
    _reg_tools_gpu.h
    ${NIFTYREG_DIR}/include
    ${NIFTYREG_DIR}-1.3/include
    ${NIFTYREG_DIR}-1.2/include
    ${NIFTYREG_DIR}-1.1/include
    ${NIFTYREG_DIR}-1.0/include
    @NIFTK_LINK_PREFIX@/include
    /usr/local/include
    /usr/include
    )

  FIND_LIBRARY(NIFTYREG_LIBRARIES
    _reg_tools_gpu
    ${NIFTYREG_DIR}/lib
    ${NIFTYREG_DIR}-1.3/lib
    ${NIFTYREG_DIR}-1.2/lib
    ${NIFTYREG_DIR}-1.1/lib
    ${NIFTYREG_DIR}-1.0/lib
    @NIFTK_LINK_PREFIX@/lib
    /usr/local/lib
    /usr/lib
    )

ELSE(CUDA_FOUND)

  FIND_PATH(NIFTYREG_INCLUDE_DIR
    _reg_tools.h
    ${NIFTYREG_DIR}/include
    ${NIFTYREG_DIR}-1.3/include
    ${NIFTYREG_DIR}-1.2/include
    ${NIFTYREG_DIR}-1.1/include
    ${NIFTYREG_DIR}-1.0/include
    @NIFTK_LINK_PREFIX@/include
    /usr/local/include
    /usr/include
    )

  FIND_LIBRARY(NIFTYREG_LIBRARIES
    _reg_tools
    ${NIFTYREG_DIR}/lib
    ${NIFTYREG_DIR}-1.3/lib
    ${NIFTYREG_DIR}-1.2/lib
    ${NIFTYREG_DIR}-1.1/lib
    ${NIFTYREG_DIR}-1.0/lib
    @NIFTK_LINK_PREFIX@/lib
    /usr/local/lib
    /usr/lib
    )

ENDIF(CUDA_FOUND)

IF(NIFTYREG_LIBRARIES AND NIFTYREG_INCLUDE_DIR)
  SET(NIFTYREG_FOUND 1)
ENDIF(NIFTYREG_LIBRARIES AND NIFTYREG_INCLUDE_DIR)
