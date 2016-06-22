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

set(ProtoBuf_FOUND)

set(ProtoBuf_DIR @ProtoBuf_DIR@ CACHE PATH "Directory containing ProtoBuf installation")

find_path(ProtoBuf_INCLUDE_DIR
  NAME protobuf.h
  PATHS ${ProtoBuf_DIR}/include
  NO_DEFAULT_PATH
)

set(ProtoBuf_LIBRARY)

foreach (LIB protobuf protobuf-lite protoc)

  if(${CMAKE_BUILD_TYPE} STREQUAL "Release")

    find_library(ProtoBuf_LIBRARY_${LIB} ${LIB}
                 PATHS ${ProtoBuf_LIBRARY_DIR}
                 PATH_SUFFIXES Release
                 NO_DEFAULT_PATH)  

  elseif(${CMAKE_BUILD_TYPE} STREQUAL "Debug")

    find_library(ProtoBuf_LIBRARY_${LIB} ${LIB}d
                 PATHS ${ProtoBuf_LIBRARY_DIR}
                 PATH_SUFFIXES Debug
                 NO_DEFAULT_PATH)

  endif()

  set(ProtoBuf_LIBRARY ${ProtoBuf_LIBRARY};${ProtoBuf_LIBRARY_${LIB}})

endforeach()

find_program(ProtoBuf_PROTOC_EXECUTABLE
  NAME protoc
  PATHS ${ProtoBuf_DIR}/bin
  NO_DEFAULT_PATH)

message( "FindProtoBuf.cmake ProtoBuf_INCLUDE_DIR:    ${ProtoBuf_INCLUDE_DIR}" )
message( "FindProtoBuf.cmake ProtoBuf_LIBRARY:        ${ProtoBuf_LIBRARY}        ${ProtoBuf_LIBRARY_DEBUG}"       )
message( "FindProtoBuf.cmake ProtoBuf_PROTOC_EXECUTABLE: ${ProtoBuf_PROTOC_EXECUTABLE}" )

if(ProtoBuf_LIBRARY AND ProtoBuf_INCLUDE_DIR AND ProtoBuf_PROTOC_EXECUTABLE)

  set(ProtoBuf_FOUND 1)

endif()


