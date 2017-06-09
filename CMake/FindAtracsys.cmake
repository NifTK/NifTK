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

if(NOT APPLE)

  find_path(Atracsys_INCLUDE_DIR
    NAMES ftkInterface.h
    PATHS "C:\\Program Files\\Atracsys\\Passive Tracking SDK\\include"
          "/opt/fusionTrack_v3_0_1_gcc-4.9/include"
    )

  set (_lib_name)
  if("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
    set(_lib_name fusionTrack64)
  else()
    set(_lib_name fusionTrack32)
  endif()

  find_library(Atracsys_LIBRARY
    NAMES ${_lib_name}
    PATHS "C:\\Program Files\\Atracsys\\Passive Tracking SDK\\lib"
          "/opt/fusionTrack_v3_0_1_gcc-4.9/lib"
  )

  if(Atracsys_INCLUDE_DIR AND Atracsys_LIBRARY)
    set(Atracsys_FOUND 1)
    message("Found Atracsys: inc=${Atracsys_INCLUDE_DIR}")
    message("Found Atracsys: lib=${Atracsys_LIBRARY}")

    get_filename_component(Atracsys_LIBRARY_DIR ${Atracsys_LIBRARY} DIRECTORY)

    get_property(_additional_search_paths GLOBAL PROPERTY MITK_ADDITIONAL_LIBRARY_SEARCH_PATHS)
    list(APPEND _additional_search_paths "${Atracsys_LIBRARY_DIR}")
    set_property(GLOBAL PROPERTY MITK_ADDITIONAL_LIBRARY_SEARCH_PATHS ${_additional_search_paths})

    message("Adding Atracsys dir: ${Atracsys_LIBRARY_DIR}")

  else()
    message("Didn't find Atracsys: inc=${Atracsys_INCLUDE_DIR}")
    message("Didn't find Atracsys: lib=${Atracsys_LIBRARY}")
  endif()

endif()

