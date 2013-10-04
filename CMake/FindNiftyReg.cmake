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


set(NIFTYREG_FOUND 0)

if(NOT NIFTYREG_DIR)
  set(NIFTYREG_DIR ${NIFTK_LINK_PREFIX}/nifty_reg CACHE PATH "Directory containing NiftyReg installation")
else(NOT NIFTYREG_DIR)
  set(NIFTYREG_DIR @NIFTYREG_DIR@ CACHE PATH "Directory containing NiftyReg installation")
endif(NOT NIFTYREG_DIR)

# disabled for now: niftyreg is never build with cuda enabled.
# so the gpu-related files will never be found.
# https://cmicdev.cs.ucl.ac.uk/trac/ticket/2764
if(FALSE AND CUDA_FOUND)

  find_path(NIFTYREG_INCLUDE_DIR
    NAME _reg_tools_gpu.h
    PATHS ${NIFTYREG_DIR}/include
    NO_DEFAULT_PATH
    )

  find_library(NIFTYREG_TOOLS_LIBRARY
    NAMES _reg_tools_gpu
    PATHS ${NIFTYREG_DIR}/lib
    NO_DEFAULT_PATH
    )

else()

  find_path(NIFTYREG_INCLUDE_DIR
    NAME _reg_tools.h
    PATHS ${NIFTYREG_DIR}/include
    NO_DEFAULT_PATH
    )

  find_library(NIFTYREG_TOOLS_LIBRARY
    NAMES _reg_tools
    PATHS ${NIFTYREG_DIR}/lib
    NO_DEFAULT_PATH
    )

endif()

if(NIFTYREG_TOOLS_LIBRARY AND NIFTYREG_INCLUDE_DIR)
  set(NIFTYREG_FOUND 1)

  set(NIFTYREG_LIBRARIES
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
    _reg_lncc
    _reg_ReadWriteImage
    reg_png
    png
    reg_nrrd
    reg_NrrdIO
    z
    )

  get_filename_component( NIFTYREG_LIBRARY_DIR ${NIFTYREG_TOOLS_LIBRARY} PATH )

  message( "NIFTYREG_INCLUDE_DIR: ${NIFTYREG_INCLUDE_DIR}" )
  message( "NIFTYREG_LIBRARY_DIR: ${NIFTYREG_LIBRARY_DIR}" )
  message( "NIFTYREG_LIBRARIES: ${NIFTYREG_LIBRARIES}" )

else(NIFTYREG_TOOLS_LIBRARY AND NIFTYREG_INCLUDE_DIR)
  message( FATAL_ERROR "ERROR: NiftyReg not Found" )
endif(NIFTYREG_TOOLS_LIBRARY AND NIFTYREG_INCLUDE_DIR)
