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

set(GFlags_FOUND)

set(GFlags_DIR @GFlags_DIR@ CACHE PATH "Directory containing GFlags installation")

set(GFlags_INCLUDE_DIR
  NAME gflags.h
  PATHS ${GFlags_DIR}/include
  NO_DEFAULT_PATH
)

set(GFlags_LIBRARY_DIR ${GFlags_DIR}/lib)

set(GFlags_LIBRARY )

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")

  find_library(GFlags_LIBRARY NAMES gflags
               PATHS ${GFlags_LIBRARY_DIR}
               PATH_SUFFIXES Release
               NO_DEFAULT_PATH)

elseif(${CMAKE_BUILD_TYPE} STREQUAL "Debug")

  find_library(GFlags_LIBRARY NAMES gflagsd
               PATHS ${GFlags_LIBRARY_DIR}
               PATH_SUFFIXES Debug
               NO_DEFAULT_PATH)

endif()

if(GFlags_LIBRARY AND GFlags_INCLUDE_DIR)

  set(GFlags_FOUND 1)

endif()

message( "GFlags_INCLUDE_DIR: ${GFlags_INCLUDE_DIR}" )
message( "GFlags_LIBRARY_DIR: ${GFlags_LIBRARY_DIR}" )
message( "GFlags_LIBRARY: ${GFlags_LIBRARY}" )
