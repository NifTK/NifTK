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

set(ProtoBuf_DIR @ProtoBuf_DIRECTORY@ CACHE PATH "Directory containing ProtoBuf installation" FORCE)

find_path(ProtoBuf_INC
  NAME message.h
  PATHS ${ProtoBuf_DIR}/include/google/protobuf
  NO_DEFAULT_PATH
)

set(ProtoBuf_LIBRARY)

foreach (LIB protobuf protoc)

  if(${CMAKE_BUILD_TYPE} STREQUAL "Release")

    find_library(ProtoBuf_LIBRARY_${LIB} ${LIB}
                 PATHS ${ProtoBuf_DIR}/lib
                 PATH_SUFFIXES Release
                 NO_DEFAULT_PATH)  

  elseif(${CMAKE_BUILD_TYPE} STREQUAL "Debug")

    find_library(ProtoBuf_LIBRARY_${LIB} ${LIB}${NIFTK_SUPERBUILD_DEBUG_POSTFIX}
                 PATHS ${ProtoBuf_DIR}/lib
                 PATH_SUFFIXES Debug
                 NO_DEFAULT_PATH)

  endif()

  set(ProtoBuf_LIBRARY ${ProtoBuf_LIBRARY};${ProtoBuf_LIBRARY_${LIB}})

endforeach()

find_program(ProtoBuf_PROTOC_EXECUTABLE
  NAME protoc
  PATHS ${ProtoBuf_DIR}/bin
  NO_DEFAULT_PATH)

if(ProtoBuf_LIBRARY AND ProtoBuf_INC AND ProtoBuf_PROTOC_EXECUTABLE)
  set(ProtoBuf_FOUND 1)
  get_filename_component(_inc_dir ${ProtoBuf_INC} PATH)
  get_filename_component(_inc_dir2 ${_inc_dir} PATH)
  set(ProtoBuf_INCLUDE_DIR ${_inc_dir2})  
endif()

message( "NifTK FindProtoBuf.cmake ProtoBuf_INCLUDE_DIR:       ${ProtoBuf_INCLUDE_DIR}" )
message( "NifTK FindProtoBuf.cmake ProtoBuf_LIBRARY:           ${ProtoBuf_LIBRARY} ${ProtoBuf_LIBRARY_DEBUG}" )
message( "NifTK FindProtoBuf.cmake ProtoBuf_PROTOC_EXECUTABLE: ${ProtoBuf_PROTOC_EXECUTABLE}" )

