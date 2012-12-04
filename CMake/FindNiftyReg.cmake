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
    NAME _reg_tools_gpu.h
    PATHS ${NIFTYREG_DIR}/include ${NIFTK_LINK_PREFIX}/include /usr/local/include /usr/include
    )

  FIND_LIBRARY(NIFTYREG_TOOLS_LIBRARY
    NAMES _reg_tools_gpu
    PATHS ${NIFTYREG_DIR}/lib ${NIFTK_LINK_PREFIX}/lib /usr/local/lib /usr/lib
    )

ELSE(CUDA_FOUND)

  FIND_PATH(NIFTYREG_INCLUDE_DIR
    NAME _reg_tools.h
    PATHS ${NIFTYREG_DIR}/include ${NIFTK_LINK_PREFIX}/include /usr/local/include /usr/include
    )

  FIND_LIBRARY(NIFTYREG_TOOLS_LIBRARY
    NAMES _reg_tools
    PATHS ${NIFTYREG_DIR}/lib ${NIFTK_LINK_PREFIX}/lib /usr/local/lib /usr/lib
    )
	
ENDIF(CUDA_FOUND)

IF(NIFTYREG_TOOLS_LIBRARY AND NIFTYREG_INCLUDE_DIR)
  SET(NIFTYREG_FOUND 1)

  SET(NIFTYREG_LIBRARIES
    _reg_KLdivergence
    _reg_blockMatching
    _reg_femTransformation
    _reg_globalTransformation
    _reg_localTransformation
    _reg_maths
    _reg_mutualinformation
    _reg_resampling
    _reg_ssd
    _reg_tools
    _reg_ReadWriteImage
    reg_png
    png
    reg_nrrd
    reg_NrrdIO
    )

  GET_FILENAME_COMPONENT( NIFTYREG_LIBRARY_DIR ${NIFTYREG_TOOLS_LIBRARY} PATH )

  MESSAGE( "NIFTYREG_INCLUDE_DIR: ${NIFTYREG_INCLUDE_DIR}" )
  MESSAGE( "NIFTYREG_LIBRARY_DIR: ${NIFTYREG_LIBRARY_DIR}" )
  MESSAGE( "NIFTYREG_LIBRARIES: ${NIFTYREG_LIBRARIES}" )

ELSE(NIFTYREG_TOOLS_LIBRARY AND NIFTYREG_INCLUDE_DIR)
  MESSAGE( FATAL_ERROR "ERROR: NiftyReg not Found" )
ENDIF(NIFTYREG_TOOLS_LIBRARY AND NIFTYREG_INCLUDE_DIR)
