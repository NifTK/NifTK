if(Boost_FOUND)
  list(APPEND ALL_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
  list(APPEND ALL_LIBRARIES ${Boost_LIBRARIES})
  link_directories(${Boost_LIBRARY_DIRS})
endif()

