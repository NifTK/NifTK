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

message( "******************* FindGFlags ******************* " )

if (NOT GFlags_FOUND)

  set(GFlags_DIR @GFlags_DIR@ CACHE PATH "Directory containing GFlags installation")

  set(GFLAGS_INCLUDE_DIR
    NAME gflags.h
    PATHS ${GFlags_DIR}/include
    NO_DEFAULT_PATH
  )

  set(GFLAGS_LIBRARY_DIR ${GFlags_DIR}/lib)

  set(GFlags_LIBRARY )

  find_library(GFlags_LIBRARY_RELEASE NAMES gflags
               PATHS ${GFLAGS_LIBRARY_DIR}
               PATH_SUFFIXES Release
               NO_DEFAULT_PATH)

  if(GFlags_LIBRARY_RELEASE)
    list(APPEND GFlags_LIBRARY ${GFlags_LIBRARY_RELEASE})
  endif()

  find_library(GFlags_LIBRARY_DEBUG NAMES gflagsd
               PATHS ${GFLAGS_LIBRARY_DIR}
               PATH_SUFFIXES Debug
               NO_DEFAULT_PATH)

  if(GFlags_LIBRARY_DEBUG)
    list(APPEND GFlags_LIBRARY ${GFlags_LIBRARY_DEBUG})
  endif()

  if(GFlags_LIBRARY AND GFLAGS_INCLUDE_DIR)

    set(GFLAGS_LIBRARY ${GFlags_LIBRARY})

    set(GFlags_FOUND 1)

  endif()

  message( "GFlags_INCLUDE_DIR: ${GFLAGS_INCLUDE_DIR}" )
  message( "GFlags_LIBRARY_DIR: ${GFlags_LIBRARY_DIR}" )
  message( "GFlags_LIBRARY: ${GFLAGS_LIBRARY}" )

endif()
