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

SET(NIFTYSIM_FOUND 0)

IF(NOT NIFTYSIM_DIR)
  SET(NIFTYSIM_DIR ${NIFTK_LINK_PREFIX}/nifty_sim CACHE PATH "Directory containing NiftySim installation")
ELSE(NOT NIFTYSIM_DIR)
  SET(NIFTYSIM_DIR @NIFTYSIM_DIR@ CACHE PATH "Directory containing NiftySim installation")
ENDIF(NOT NIFTYSIM_DIR)

IF(CUDA_FOUND)

  FIND_PATH(NIFTYSIM_INCLUDE_DIR
    tledSolverGPU.h
    ${NIFTYSIM_DIR}/include
    ${NIFTYSIM_DIR}-1.0/include
    @NIFTK_LINK_PREFIX@/include
    /usr/local/include
    /usr/include
  )
  
  FIND_LIBRARY(NIFTYSIM_LIBRARIES
    tled
    ${NIFTYSIM_DIR}/lib
    NIFTYSIM_DIR  ${NIFTYSIM_DIR}-1.0/lib
    @NIFTK_LINK_PREFIX@/lib
    /usr/local/lib
    /usr/lib
  )

ELSE(CUDA_FOUND)

  FIND_PATH(NIFTYSIM_INCLUDE_DIR
    tledSolverCPU.h
    ${NIFTYSIM_DIR}/include
    ${NIFTYSIM_DIR}-1.0/include
    @NIFTK_LINK_PREFIX@/include
    /usr/local/include
    /usr/include
  )

  FIND_LIBRARY(NIFTYSIM_LIBRARIES
    tled
    ${NIFTYSIM_DIR}/lib
    ${NIFTYSIM_DIR}-1.0/lib
    @NIFTK_LINK_PREFIX@/lib
    /usr/local/lib
    /usr/lib
  )

ENDIF(CUDA_FOUND)

IF(NIFTYSIM_LIBRARIES AND NIFTYSIM_INCLUDE_DIR)
  SET(NIFTYSIM_FOUND 1)
ENDIF(NIFTYSIM_LIBRARIES AND NIFTYSIM_INCLUDE_DIR)

