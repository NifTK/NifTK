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
    NAMES ftkInterface.h
    PATHS "C:\\Program Files\\Atracsys\\Passive Tracking SDK\\include"
    )

  set (_lib_name)
  if(${CMAKE_GENERATOR} MATCHES "Win64")
    set(_lib_name fusionTrack64)
  else()
    set(_lib_name fusionTrack32)
  endif()

  find_library(Atracsys_LIBRARY
    NAMES ${_lib_name}
    PATHS "C:\\Program Files\\Atracsys\\Passive Tracking SDK\\lib"
  )

  if(Atracsys_INCLUDE_DIR AND Atracsys_LIBRARY)
    set(Atracsys_FOUND 1)
    message("Found Atracsys: inc=${Atracsys_INCLUDE_DIR}")
    message("Found Atracsys: lib=${Atracsys_LIBRARY}")
  else()
    message("Didn't find Atracsys: inc=${Atracsys_INCLUDE_DIR}")
    message("Didn't find Atracsys: lib=${Atracsys_LIBRARY}")
  endif()

endif(WIN32)

