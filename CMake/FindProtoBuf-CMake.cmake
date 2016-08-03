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

set(ProtoBuf-CMake_FOUND)

set(ProtoBuf-CMake_DIR @ProtoBuf-CMake_DIR@ CACHE PATH "Directory containing ProtoBuf-CMake installation")

find_file(ProtoBuf-CMake_CMakeLists
  NAME CMakeLists.txt
  PATHS ${ProtoBuf-CMake_SOURCE_DIR}
  NO_DEFAULT_PATH
)

message( "FindProtoBuf-CMake.cmake ProtoBuf-CMake_CMakeLists: ${ProtoBuf-CMake_CMakeLists}" )

if(ProtoBuf-CMake_CMakeLists)

  set(ProtoBuf-CMake_FOUND 1)

endif()


