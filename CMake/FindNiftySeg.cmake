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


if (NOT NiftySeg_FOUND)

  set(NiftySeg_DIR @NiftySeg_DIR@ CACHE PATH "Directory containing NiftySeg installation")

  find_path(NiftySeg_INCLUDE_DIR
    NAME _seg_tools.h
    PATHS ${NiftySeg_DIR}/include
    NO_DEFAULT_PATH
  )

  find_library(NiftySeg_LIBRARIES
    NAMES _seg_nifti _seg_nifti${NiftySeg_DEBUG_POSTFIX}
    PATHS ${NiftySeg_DIR}/lib
    NO_DEFAULT_PATH
  )

  if(NiftySeg_LIBRARIES AND NiftySeg_INCLUDE_DIR)

    set(NiftySeg_FOUND 1)

    foreach (_library
        _seg_EM
        _seg_FMM
        _seg_LabFusion
        _seg_LoAd
        _seg_nifti
        _seg_tools
        z
      )

      set(NiftySeg_LIBRARIES ${NiftySeg_LIBRARIES} ${_library}${NiftySeg_DEBUG_POSTFIX})

    endforeach()

    message( "NiftySeg_INCLUDE_DIR: ${NiftySeg_INCLUDE_DIR}" )
    message( "NiftySeg_LIBRARY_DIR: ${NiftySeg_LIBRARY_DIR}" )
    message( "NiftySeg_LIBRARIES: ${NiftySeg_LIBRARIES}" )

  endif()

endif()
