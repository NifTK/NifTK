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

if (NOT NiftyReg_FOUND)

  set(NiftyReg_DIR @NiftyReg_DIR@ CACHE PATH "Directory containing NiftyReg installation" FORCE)

  # disabled for now: niftyreg is never build with cuda enabled.
  # so the gpu-related files will never be found.
  # https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK/issues/2764
  if(FALSE AND CUDA_FOUND)

    find_path(NiftyReg_INCLUDE_DIR
      NAME _reg_tools_gpu.h
      PATHS ${NiftyReg_DIR}/include
      NO_DEFAULT_PATH
    )

    find_library(NiftyReg_TOOLS_LIBRARY
      NAMES _reg_tools_gpu _reg_tools_gpu${NiftyReg_DEBUG_POSTFIX}
      PATHS ${NiftyReg_DIR}/lib
      NO_DEFAULT_PATH
    )

  else()

    find_path(NiftyReg_INCLUDE_DIR
      NAME _reg_tools.h
      PATHS ${NiftyReg_DIR}/include
      NO_DEFAULT_PATH
    )

    find_library(NiftyReg_TOOLS_LIBRARY
      NAMES _reg_tools _reg_tools${NiftyReg_DEBUG_POSTFIX}
      PATHS ${NiftyReg_DIR}/lib
      NO_DEFAULT_PATH
    )

  endif()

  if(NiftyReg_TOOLS_LIBRARY AND NiftyReg_INCLUDE_DIR)
    set(NiftyReg_FOUND 1)

    foreach (_library
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

      find_library(_library_with_postfix
        NAMES ${_library} ${_library}${NiftyReg_DEBUG_POSTFIX}
        PATHS ${NiftyReg_DIR}/lib
        NO_DEFAULT_PATH
      )

      set(NiftyReg_LIBRARIES ${NiftyReg_LIBRARIES} ${_library_with_postfix}})

    endforeach()

    get_filename_component( NiftyReg_LIBRARY_DIR ${NiftyReg_TOOLS_LIBRARY} PATH )

    message( "NiftyReg_INCLUDE_DIR: ${NiftyReg_INCLUDE_DIR}" )
    message( "NiftyReg_LIBRARY_DIR: ${NiftyReg_LIBRARY_DIR}" )
    message( "NiftyReg_LIBRARIES: ${NiftyReg_LIBRARIES}" )

  else()
    message( FATAL_ERROR "ERROR: NiftyReg not Found" )
  endif()

endif()
