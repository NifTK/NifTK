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

set(HDF5_FOUND)

set(HDF5_DIR @HDF5_DIRECTORY@ CACHE PATH "Directory containing HDF5 installation" FORCE)

find_path(HDF5_INCLUDE_DIR
  NAME hdf5.h
  PATHS ${HDF5_DIR}/include
  NO_DEFAULT_PATH
)

set(HDF5_LIBRARY_DIR ${HDF5_DIR}/lib)
set(HDF5_LIBRARIES )

set(HDF5_PREFIX niftk)

# We set the CMAKE_DEBUG_POSTFIX to 'd' in SuperBuild.cmake.
# But when HDF5 switches to Debug, you get _debug on *nix and
# _D on Windows, and I've not been able to change this.

set(HDF5_SUFFIX)
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  if(WIN32)
    set(HDF5_SUFFIX _D${NIFTK_SUPERBUILD_DEBUG_POSTFIX})
  else()
    set(HDF5_SUFFIX _debug${NIFTK_SUPERBUILD_DEBUG_POSTFIX})
  endif()
endif()

# message("NifTK FindHDF5.cmake Looking for prefix=${HDF5_PREFIX}, suffix=${HDF5_SUFFIX}")

foreach (LIB
         hdf5
         hdf5_tools
         hdf5_hl
         hdf5_cpp
         hdf5_hl_cpp)

  if(${CMAKE_BUILD_TYPE} STREQUAL "Release")

    set(HDF5_LIBRARY_RELEASE_${LIB} )

    find_library(HDF5_LIBRARY_RELEASE_${LIB} NAME ${HDF5_PREFIX}${LIB}${HDF5_SUFFIX}
                 PATHS ${HDF5_LIBRARY_DIR}
                 PATH_SUFFIXES Release
                 NO_DEFAULT_PATH)
             
    if(HDF5_LIBRARY_RELEASE_${LIB})
      set(HDF5_LIBRARIES ${HDF5_LIBRARIES};${HDF5_LIBRARY_RELEASE_${LIB}})
    endif()

  endif()

  if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")

    set(HDF5_LIBRARY_DEBUG_${LIB} )

    find_library(HDF5_LIBRARY_DEBUG_${LIB} NAME ${HDF5_PREFIX}${LIB}${HDF5_SUFFIX} 
                 PATHS ${HDF5_LIBRARY_DIR}
                 PATH_SUFFIXES Debug
                 NO_DEFAULT_PATH)

    if(HDF5_LIBRARY_DEBUG_${LIB})
      set(HDF5_LIBRARIES ${HDF5_LIBRARIES};${HDF5_LIBRARY_DEBUG_${LIB}})
    endif()

  endif()

endforeach()

if(HDF5_LIBRARIES AND HDF5_INCLUDE_DIR)
  set(HDF5_FOUND 1)
endif()

message( "NifTK FindHDF5.cmake: HDF5_INCLUDE_DIR: ${HDF5_INCLUDE_DIR}" )
message( "NifTK FindHDF5.cmake: HDF5_LIBRARY_DIR: ${HDF5_LIBRARY_DIR}" )
message( "NifTK FindHDF5.cmake: HDF5_LIBRARIES:   ${HDF5_LIBRARIES}" )

