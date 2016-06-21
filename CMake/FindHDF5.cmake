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

message("FindHDF5.cmake")

if (NOT HDF5_FOUND)

  set(HDF5_DIR @HDF5_DIR@ CACHE PATH "Directory containing HDF5 installation")

  find_path(HDF5_INCLUDE_DIR
    NAME hdf5.h
    PATHS ${HDF5_DIR}/include
    NO_DEFAULT_PATH
  )

  set(HDF5_LIBRARY_DIR ${HDF5_DIR}/lib)
  set(HDF5_LIBRARIES )

  foreach (LIB
           hdf5
           hdf5_tools
           hdf5_hl
           hdf5_cpp
           hdf5_hl_cpp)

    message("HDF5.cmake Looking for ${LIB}")


    if(${CMAKE_BUILD_TYPE} STREQUAL "Release")

      set(HDF5_LIBRARY_RELEASE_${LIB} )

      find_library(HDF5_LIBRARY_RELEASE_${LIB} NAME ${LIB}
                   PATHS ${HDF5_LIBRARY_DIR}
                   PATH_SUFFIXES Release
                   NO_DEFAULT_PATH)
               
      if(HDF5_LIBRARY_RELEASE_${LIB})
        message("HDF5.cmake Found ${HDF5_LIBRARY_RELEASE_${LIB}}")
        set(HDF5_LIBRARIES ${HDF5_LIBRARIES};${HDF5_LIBRARY_RELEASE_${LIB}})
      endif()

    endif()

    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")

      set(HDF5_LIBRARY_DEBUG_${LIB} )

      find_library(HDF5_LIBRARY_DEBUG_${LIB} NAME ${LIB}d
                   PATHS ${HDF5_LIBRARY_DIR}
                   PATH_SUFFIXES Debug
                   NO_DEFAULT_PATH)

      if(HDF5_LIBRARY_DEBUG_${LIB})
        message("HDF5.cmake Found ${HDF5_LIBRARY_DEBUG_${LIB}}")
        set(HDF5_LIBRARIES ${HDF5_LIBRARIES};${HDF5_LIBRARY_DEBUG_${LIB}})
      endif()

    endif()

  endforeach()

  if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")

    foreach (LIB
             hdf5_debug
             hdf5_tools_debug
             hdf5_hl_debug
             hdf5_cpp_debug
             hdf5_hl_cpp_debug)

      message("HDF5.cmake Looking for ${LIB}")

      set(HDF5_LIBRARY_DEBUG_${LIB} )

      find_library(HDF5_LIBRARY_DEBUG_${LIB} NAME ${LIB}
                   PATHS ${HDF5_LIBRARY_DIR}
                   PATH_SUFFIXES Debug
                   NO_DEFAULT_PATH)

      if(HDF5_LIBRARY_DEBUG_${LIB})
        message("HDF5.cmake Found ${HDF5_LIBRARY_DEBUG_${LIB}}")
        set(HDF5_LIBRARIES ${HDF5_LIBRARIES};${HDF5_LIBRARY_DEBUG_${LIB}})
      endif()

    endforeach()

  endif()


  if(HDF5_LIBRARIES AND HDF5_INCLUDE_DIR)

    set(HDF5_FOUND 1)

    set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})

    message( "HDF5_INCLUDE_DIR: ${HDF5_INCLUDE_DIR}" )
    message( "HDF5_LIBRARY_DIR: ${HDF5_LIBRARY_DIR}" )
    message( "HDF5_LIBRARIES:   ${HDF5_LIBRARIES}" )

  endif()

endif()
