IF(NiftyLink_FOUND)
  list(APPEND ALL_INCLUDE_DIRECTORIES ${NiftyLink_INCLUDE_DIRS})
  list(APPEND ALL_LIBRARIES ${NiftyLink_LIBRARIES})
  link_directories(${NiftyLink_LIBRARY_DIRS})
ENDIF()

