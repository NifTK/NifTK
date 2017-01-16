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

set(UltrasonixUlterius_FOUND 0)

if(WIN32)

  if(${CMAKE_GENERATOR} MATCHES "Win64")
    message("Cannot build niftkUltrasonixDataSourceService in 64 bit mode")
  else()

    find_path(Ultrasonix_ULTERIUS_INCLUDE_DIR
      NAMES ulterius.h

      # It's a 32 bit library
      PATHS "C:/ultrasonix/sdk_612/ulterius/inc"
            "C:/Program Files (x86)/sdk_612/ulterius/inc"
            "C:/Program Files (x86)/sdk_6.0.4_(00.036.203)/ulterius/inc"
      )

    find_library(Ultrasonix_ULTERIUS_LIBRARY
      NAMES ulterius_old

      # It's a 32 bit library
      PATHS "C:/ultrasonix/sdk_612/ulterius/lib"
            "C:/Program Files (x86)/sdk_612/ulterius/lib"
            "C:/Program Files (x86)/sdk_6.0.4_(00.036.203)/ulterius/lib"
    )

    if(Ultrasonix_ULTERIUS_INCLUDE_DIR AND Ultrasonix_ULTERIUS_LIBRARY)

      get_filename_component(Ultrasonix_ULTERIUS_LIBRARY_DIR ${Ultrasonix_ULTERIUS_LIBRARY} DIRECTORY)
      set(Ultrasonix_ULTERIUS_BIN_DIR "${Ultrasonix_ULTERIUS_LIBRARY_DIR}/../../bin")

      get_property(_additional_search_paths GLOBAL PROPERTY MITK_ADDITIONAL_LIBRARY_SEARCH_PATHS)
      list(APPEND _additional_search_paths "${Ultrasonix_ULTERIUS_BIN_DIR}")
      set_property(GLOBAL PROPERTY MITK_ADDITIONAL_LIBRARY_SEARCH_PATHS ${_additional_search_paths})

      set(UltrasonixUlterius_FOUND 1)

    endif()
  endif()
endif(WIN32)

