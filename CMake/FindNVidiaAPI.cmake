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
  
set(NVAPI_FOUND 0)

# Note: This is ONLY valid On Windows
if(WIN32)

  set(NVAPI_POSSIBLE_INCLUDE_PATHS 
    "C:/Program Files/nvapi/"
    "C:/Program Files/R304-developer/"
    "E:/build/nvapi/"
    "E:/build/R304-developer/"
    "E:/NifTK/nvapi/"
    "E:/NifTK/R304-developer/"
    ${CMAKE_SOURCE_DIR}/../nvapi/
  )

  find_path(NVAPI_INCLUDE_DIR
    NAMES nvapi.h
    PATHS ${NVAPI_POSSIBLE_INCLUDE_PATHS}
  )

  if("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
    set(NVAPI_ARCH_DIR amd64)
    set(NVAPI_LIBRARY_NAME nvapi64)
  else()
    set(NVAPI_ARCH_DIR x86)
    set(NVAPI_LIBRARY_NAME nvapi)
  endif()

  set(NVAPI_POSSIBLE_LIBRARY_PATHS
    ${NVAPI_INCLUDE_DIR}/${NVAPI_ARCH_DIR}
  )

  find_library(NVAPI_LIBRARY
    NAMES ${NVAPI_LIBRARY_NAME}
    HINTS ${NVAPI_POSSIBLE_LIBRARY_PATHS} 
  )

  if(NVAPI_INCLUDE_DIR AND NVAPI_LIBRARY)
    set(NVAPI_FOUND 1)
  endif()

# Note: This is ONLY valid On Windows
endif(WIN32)

