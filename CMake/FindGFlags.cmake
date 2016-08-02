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

if (NOT GFlags_FOUND)

  set(GFlags_DIR @GFlags_DIR@ CACHE PATH "Directory containing GFlags installation")

  find_path(GFlags_INCLUDE_DIR
    NAME gflags.h
    PATHS ${GFlags_DIR}/include
    NO_DEFAULT_PATH
  )

  find_library(GFlags_LIBRARY
    NAMES gflags gflags{GFlags_DEBUG_POSTFIX}
    PATHS ${GFlags_DIR}/lib
    NO_DEFAULT_PATH
  )

  if(GFlags_LIBRARY AND GFlags_INCLUDE_DIR)

    set(GFlags_FOUND 1)

    foreach (_library
        gflags
        gflags_nothreads
      )

      set(GFlags_LIBRARY ${GFlags_LIBRARY} ${_library}${GFlags_DEBUG_POSTFIX})

    endforeach()

    message( "GFlags_INCLUDE_DIR: ${GFlags_INCLUDE_DIR}" )
    message( "GFlags_LIBRARY_DIR: ${GFlags_LIBRARY_DIR}" )
    message( "GFlags_LIBRARY: ${GFlags_LIBRARY}" )

  endif()

endif()
