#-----------------------------------------------------------------------------
# OpenCV
#-----------------------------------------------------------------------------


# Sanity checks
if(DEFINED OpenCV_DIR AND NOT EXISTS ${OpenCV_DIR})
  message(FATAL_ERROR "OpenCV_DIR variable is defined but corresponds to non-existing directory")
endif()

set(proj OpenCV)
set(proj_DEPENDENCIES)
set(OpenCV_DEPENDS ${proj})

if(NOT DEFINED OpenCV_DIR)

#    list(APPEND additional_cmake_args 
#	  -DWITH_QT_OPENGL:BOOL=OFF
#	  -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
#    )

  # same as mitk's
  set(opencv_url http://mitk.org/download/thirdparty/OpenCV-2.4.2.tar.bz2)
  set(opencv_url_md5 d5d13c4a65dc96cdfaad54767e428215)

  
  ExternalProject_Add(${proj}
    URL ${opencv_url}
    URL_MD5 ${opencv_url_md5}
    BINARY_DIR ${proj}-build
    UPDATE_COMMAND  ""
    INSTALL_COMMAND ""
    CMAKE_GENERATOR ${GEN}
    CMAKE_CACHE_ARGS
    ${EP_COMMON_ARGS}
    -DBUILD_DOCS:BOOL=OFF
    -DBUILD_TESTS:BOOL=OFF
    -DBUILD_EXAMPLES:BOOL=OFF
    -DBUILD_DOXYGEN_DOCS:BOOL=OFF
    -DWITH_CUDA:BOOL=ON
    -DWITH_QT:BOOL=OFF
    -DADDITIONAL_C_FLAGS:STRING=${NIFTK_ADDITIONAL_C_FLAGS}
    -DADDITIONAL_CXX_FLAGS:STRING=${NIFTK_ADDITIONAL_CXX_FLAGS}
    DEPENDS ${proj_DEPENDENCIES}
  )
  SET(OpenCV_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)
  MESSAGE("SuperBuild loading OpenCV from ${OpenCV_DIR}")

	
 #   ExternalProject_Add(${proj}
 #     SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj}-src
 #     BINARY_DIR ${proj}-build
 #     PREFIX ${proj}-cmake
 #     URL ${opencv_url}
 #     URL_MD5 ${opencv_url_md5}
 #     INSTALL_COMMAND ""
 #     CMAKE_GENERATOR ${gen}
 #     CMAKE_ARGS
 #       ${ep_common_args}
 #       -DBUILD_DOCS:BOOL=OFF
 #       -DBUILD_TESTS:BOOL=OFF
 #       -DBUILD_EXAMPLES:BOOL=OFF
 #       -DBUILD_DOXYGEN_DOCS:BOOL=OFF
 #       -DWITH_CUDA:BOOL=ON
#		-DWITH_QT:BOOL=OFF
#        ${additional_cmake_args}
#      DEPENDS ${proj_DEPENDENCIES}
#    )
#    set(OpenCV_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)

else()

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif()
