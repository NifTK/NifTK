#/*================================================================================
#
#  NiftyGuide: A suite of utilities for Image Guided Interventions.
#
#  Copyright (c) 2013 University College London (UCL).
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#================================================================================*/

set(UltrasonixUlterius_FOUND 0)

if(WIN32)

  if(${CMAKE_GENERATOR} MATCHES "Win64")
    message("Cannot build niftkUltrasonixDataSourceService in 64 bit mode")
  else()

    find_path(Ultrasonix_ULTERIUS_INCLUDE_DIR
      NAMES ulterius.h

      # It's a 32 bit library
      PATHS "C:/Program Files (x86)/sdk_6.0.4_(00.036.203)/ulterius/inc"
      )

    find_library(Ultrasonix_ULTERIUS_LIBRARY
      NAMES ulterius_old

      # It's a 32 bit library
      PATHS "C:/Program Files (x86)/sdk_6.0.4_(00.036.203)/ulterius/lib"
    )

    if(Ultrasonix_ULTERIUS_INCLUDE_DIR AND Ultrasonix_ULTERIUS_LIBRARY)

      get_filename_component(Ultrasonix_ULTERIUS_LIBRARY_DIR ${Ultrasonix_ULTERIUS_LIBRARY} DIRECTORY)
      set(Ultrasonix_ULTERIUS_BIN_DIR "${Ultrasonix_ULTERIUS_LIBRARY_DIR}/../../bin")

      get_property(_additional_search_paths GLOBAL PROPERTY MITK_ADDITIONAL_LIBRARY_SEARCH_PATHS)
      list(APPEND _additional_search_paths ${Ultrasonix_ULTERIUS_BIN_DIR})
      set_property(GLOBAL PROPERTY MITK_ADDITIONAL_LIBRARY_SEARCH_PATHS ${_additional_search_paths})

      set(UltrasonixUlterius_FOUND 1)

    endif()
  endif()
endif(WIN32)

