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

set(GLog_FOUND)

set(GLog_DIR @GLog_DIR@ CACHE PATH "Directory containing GLog installation")

message( FATAL_ERROR "FindGLog.cmake CMAKE_DEBUG_POSTFIX: ${CMAKE_DEBUG_POSTFIX}" )


set(GLog_INCLUDE_DIR
  NAME glog.h
  PATHS ${GLog_DIR}/include
  NO_DEFAULT_PATH
)

set(GLog_LIBRARY_DIR ${GLog_DIR}/lib)

set(GLog_LIBRARY )

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")

  find_library(GLog_LIBRARY NAMES glog
               PATHS ${GLog_LIBRARY_DIR}
               PATH_SUFFIXES Release
               NO_DEFAULT_PATH)

elseif(${CMAKE_BUILD_TYPE} STREQUAL "Debug")

  find_library(GLog_LIBRARY NAMES glogd
               PATHS ${GLog_LIBRARY_DIR}
               PATH_SUFFIXES Debug
               NO_DEFAULT_PATH)

endif()

if(GLog_LIBRARY AND GLog_INCLUDE_DIR)

  set(GLog_FOUND 1)

endif()

message( "GLog_INCLUDE_DIR: ${GLog_INCLUDE_DIR}" )
message( "GLog_LIBRARY_DIR: ${GLog_LIBRARY_DIR}" )
message( "GLog_LIBRARY:     ${GLog_LIBRARY}" )
