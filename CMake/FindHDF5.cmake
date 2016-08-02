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

if (NOT HDF5_FOUND)

  set(HDF5_DIR @HDF5_DIR@ CACHE PATH "Directory containing HDF5 installation")

  find_path(HDF5_INCLUDE_DIR
    NAME hdf5.h
    PATHS ${HDF5_DIR}/include
    NO_DEFAULT_PATH
  )

  find_library(HDF5_LIBRARY
    NAMES hdf5 hdf5{HDF5_DEBUG_POSTFIX}
    PATHS ${HDF5_DIR}/lib
    NO_DEFAULT_PATH
  )

  if(HDF5_LIBRARY AND HDF5_INCLUDE_DIR)

    set(HDF5_FOUND 1)

    foreach (_library
      hdf5_debug
      hdf5_tools_debug
      hdf5_hl_debug
      hdf5_cpp_debug
      hdf5_hl_cpp_debug
      )

      set(HDF5_LIBRARIES ${HDF5_LIBRARIES} ${_library}${HDF5_DEBUG_POSTFIX})

    endforeach()

    message( "HDF5_INCLUDE_DIR: ${HDF5_INCLUDE_DIR}" )
    message( "HDF5_LIBRARY_DIR: ${HDF5_LIBRARY_DIR}" )
    message( "HDF5_LIBRARIES: ${HDF5_LIBRARIES}" )

  endif()

endif()
