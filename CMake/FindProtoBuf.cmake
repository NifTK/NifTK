
if (NOT ProtoBuf_FOUND)

  set(ProtoBuf_DIR @ProtoBuf_DIR@ CACHE PATH "Directory containing ProtoBuf installation")

  find_path(ProtoBuf_INCLUDE_DIR
    NAME protobuf.h
    PATHS ${ProtoBuf_DIR}/include
    NO_DEFAULT_PATH
  )

  find_library(ProtoBuf_LIBRARY
    NAME protobuf
    PATHS ${ProtoBuf_DIR}/lib
    NO_DEFAULT_PATH
  )
  find_library(ProtoBuf_LIBRARY_DEBUG
    NAME protobuf{ProtoBuf_DEBUG_POSTFIX}
    PATHS ${ProtoBuf_DIR}/lib
    NO_DEFAULT_PATH
  )

  find_library(ProtoBuf_LITE_LIBRARY
    NAME protobuf-lite
    PATHS ${ProtoBuf_DIR}/lib
    NO_DEFAULT_PATH
  )
  find_library(ProtoBuf_LITE_LIBRARY_DEBUG
    NAME protobuf-lite{ProtoBuf_DEBUG_POSTFIX}
    PATHS ${ProtoBuf_DIR}/lib
    NO_DEFAULT_PATH
  )

  find_library(ProtoBuf_PROTOC_LIBRARY
    NAME protoc
    PATHS ${ProtoBuf_DIR}/lib
    NO_DEFAULT_PATH
  )
  find_library(ProtoBuf_PROTOC_LIBRARY_DEBUG
    NAME protoc{ProtoBuf_DEBUG_POSTFIX}
    PATHS ${ProtoBuf_DIR}/lib
    NO_DEFAULT_PATH
  )

  find_program(ProtoBuf_PROTOC_EXECUTABLE
    NAME protoc
    PATHS ${ProtoBuf_DIR}/bin
    NO_DEFAULT_PATH
  )

  if(ProtoBuf_LIBRARY OR ProtoBuf_LIBRARY_DEBUG AND ProtoBuf_INCLUDE_DIR AND ProtoBuf_PROTOC_EXECUTABLE)

    set(ProtoBuf_FOUND 1)

    message( "ProtoBuf_INCLUDE_DIR: ${ProtoBuf_INCLUDE_DIR}" )
    message( "ProtoBuf_LIBRARY:        ${ProtoBuf_LIBRARY}        ${ProtoBuf_LIBRARY_DEBUG}"       )
    message( "ProtoBuf_LITE_LIBRARY:   ${ProtoBuf_LITE_LIBRARY}   ${ProtoBuf_LITE_LIBRARY_DEBUG}"  )
    message( "ProtoBuf_PROTOC_LIBRARY: ${ProtoBuf_PROTOC_LIBRARY} ${ProtoBuf_PROTOC_LIBRARY_DEBUG}")
    message( "ProtoBuf_PROTOC_EXECUTABLE: ${ProtoBuf_PROTOC_EXECUTABLE}" )

  endif()

endif()


include(${CMAKE_CURRENT_LIST_DIR}/GenericFindModuleConfig.cmake)
set(PROTOBUF_LIBRARIES libprotobuf)
set(PROTOBUF_LITE_LIBRARIES libprotobuf-lite)
set(PROTOBUF_PROTOC_LIBRARIES libprotoc)