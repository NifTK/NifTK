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

set(OpenBLAS_FOUND)

set(OpenBLAS_DIR @OpenBLAS_DIR@ CACHE PATH "Directory containing OpenBLAS installation")

set(OpenBLAS_INCLUDE_DIR
  NAME cblas.h
  PATHS ${OpenBLAS_DIR}
  NO_DEFAULT_PATH
)

set(OpenBLAS_LIBRARY_DIR ${OpenBLAS_DIR}/lib)

set(OpenBLAS_LIBRARY )

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")

  find_library(OpenBLAS_LIBRARY NAMES openblas
               PATHS ${OpenBLAS_LIBRARY_DIR}
               PATH_SUFFIXES Release
               NO_DEFAULT_PATH)

elseif(${CMAKE_BUILD_TYPE} STREQUAL "Debug")

  find_library(OpenBLAS_LIBRARY NAMES openblasd
               PATHS ${OpenBLAS_LIBRARY_DIR}
               PATH_SUFFIXES Debug
               NO_DEFAULT_PATH)

endif()

if(OpenBLAS_LIBRARY AND OpenBLAS_INCLUDE_DIR)

  set(OpenBLAS_FOUND 1)

endif()

message( "OpenBLAS_INCLUDE_DIR: ${OpenBLAS_INCLUDE_DIR}" )
message( "OpenBLAS_LIBRARY_DIR: ${OpenBLAS_LIBRARY_DIR}" )
message( "OpenBLAS_LIBRARY:     ${OpenBLAS_LIBRARY}" )
