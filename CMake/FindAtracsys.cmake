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
  message("Found Atracsys: inc=${Atracsys_INCLUDE_DIR}")

  if("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
    set(_lib_name fusionTrack64)
    set(_device_lib_name device64)
  else()
    set(_lib_name fusionTrack32)
    set(_device_lib_name device32)
  endif()

  if(WIN32)
    set(_previous_suffix ${CMAKE_FIND_LIBRARY_SUFFIXES})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".dll")
    find_library(Atracsys_DLL
      NAMES ${_lib_name}
      PATHS "C:/Windows/System32"
            "C:/Windows/SysWOW64"
    )
    find_library(Atracsys_DEVICE_DLL
      NAMES ${_device_lib_name}
      PATHS "C:/Windows/System32"
            "C:/Windows/SysWOW64"
    )
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_previous_suffix})
    message("Found Atracsys: dll=${Atracsys_DLL}")
    message("Found Atracsys: device=${Atracsys_DEVICE_DLL}")
  endif()

  find_library(Atracsys_LIBRARY
    NAMES ${_lib_name}
    PATHS "C:\\Program Files\\Atracsys\\Passive Tracking SDK\\lib"
          "/opt/fusionTrack_v3_0_1_gcc-4.9/lib"
  )
  message("Found Atracsys: lib=${Atracsys_LIBRARY}")

  set(Atracsys_FOUND 0)

  if(Atracsys_INCLUDE_DIR AND Atracsys_LIBRARY)
    if(WIN32)
      if(Atracsys_DLL AND Atracsys_DEVICE_DLL)
        set(Atracsys_FOUND 1)
        set(_atracsys_lib ${Atracsys_DLL})
        list(APPEND CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS ${Atracsys_DLL})
        list(APPEND CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS ${Atracsys_DEVICE_DLL})
      endif()
    else()
      set(Atracsys_FOUND 1)
      set(_atracsys_lib ${Atracsys_LIBRARY})
    endif()

    if (_atracsys_lib)
      get_filename_component(Atracsys_LIBRARY_DIR ${_atracsys_lib} DIRECTORY)
      get_property(_additional_search_paths GLOBAL PROPERTY MITK_ADDITIONAL_LIBRARY_SEARCH_PATHS)
      list(APPEND _additional_search_paths "${Atracsys_LIBRARY_DIR}")
      set_property(GLOBAL PROPERTY MITK_ADDITIONAL_LIBRARY_SEARCH_PATHS ${_additional_search_paths})
      message("Adding Atracsys dir: ${Atracsys_LIBRARY_DIR}")
    endif()

  endif()

  if(Atracsys_FOUND)
    message("Found Atracsys")
  else()
    message("Didn't find Atracsys. If you are compiling 64 bit NifTK, check your CMake is 64 bit.")
  endif()

endif()

