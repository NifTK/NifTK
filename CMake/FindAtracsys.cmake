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

set(Atracsys_FOUND 0)

if(WIN32)

  find_path(Atracsys_INCLUDE_DIR
    NAMES blah.h
    PATHS "C:/Program Files/Atracsys/inc"
    )

  set (_lib_name)
  if(${CMAKE_GENERATOR} MATCHES "Win64")
    set(_lib_name blah)
  else()
    set(_lib_name blah)
  endif()

  find_library(Atracsys_LIBRARY
    NAMES ${_lib_name}
    PATHS "C:/Program Files/Atracsys/lib"
  )

  if(Atracsys_INCLUDE_DIR AND Atracsys_LIBRARY)
    set(Atracsys_FOUND 1)
  endif()

endif(WIN32)
set(Atracsys_FOUND 1)
