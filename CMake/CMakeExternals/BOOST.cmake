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

#-----------------------------------------------------------------------------
# BOOST. In SuperBuild we do install it to get include and lib folders
#        in standard place. Which then means that FindBoost will find it.
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED BOOST_ROOT AND NOT EXISTS ${BOOST_ROOT})
  message(FATAL_ERROR "BOOST_ROOT variable is defined but corresponds to non-existing directory \"${BOOST_ROOT}\".")
endif()

set(proj BOOST)
set(proj_DEPENDENCIES )
set(proj_INSTALL ${EP_BASE}/Install/${proj})
set(BOOST_DEPENDS ${proj})

if(NOT DEFINED BOOST_ROOT)

  set(additional_cmake_args )

  set(BOOST_ARGS
    -DENABLE_MULTI_THREADED:BOOL=ON
    -DENABLE_SINGLE_THREADED:BOOL=OFF
    -DWITH_MPI:BOOL=OFF
  )

  if(${BUILD_SHARED_LIBS})
    if (BUILD_MESHING)
      if (NOT WIN32) 
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
      endif (NOT WIN32)
      set(BOOST_ARGS
        ${BOOST_ARGS}
       -DENABLE_SHARED:BOOL=ON
       -DBUILD_SHARED_LIBS:BOOL=ON
       -DENABLE_STATIC:BOOL=ON
       -DENABLE_STATIC_RUNTIME:BOOL=OFF
      )
    else (BUILD_MESHING)
      set(BOOST_ARGS
        ${BOOST_ARGS}
        -DENABLE_SHARED:BOOL=ON
        -DBUILD_SHARED_LIBS:BOOL=ON
        -DENABLE_STATIC:BOOL=OFF
        -DENABLE_STATIC_RUNTIME:BOOL=OFF
      )
    endif (BUILD_MESHING)
  else(${BUILD_SHARED_LIBS})
    if (NOT WIN32)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif (NOT WIN32)
    set(BOOST_ARGS
      ${BOOST_ARGS}
      -DENABLE_SHARED:BOOL=OFF
      -DBUILD_SHARED_LIBS:BOOL=OFF
      -DENABLE_STATIC:BOOL=ON
      -DENABLE_STATIC_RUNTIME:BOOL=ON
      )
  endif(${BUILD_SHARED_LIBS})

  if(${CMAKE_BUILD_TYPE} MATCHES "Release")
    set(BOOST_ARGS
      ${BOOST_ARGS}
      -DENABLE_RELEASE:BOOL=ON
      -DENABLE_DEBUG:BOOL=OFF
    )
  else(${CMAKE_BUILD_TYPE} MATCHES "Release")
    set(BOOST_ARGS
      ${BOOST_ARGS}
      -DENABLE_RELEASE:BOOL=OFF
      -DENABLE_DEBUG:BOOL=ON
    )
  endif(${CMAKE_BUILD_TYPE} MATCHES "Release")

  niftkMacroGetChecksum(NIFTK_CHECKSUM_BOOST ${NIFTK_LOCATION_BOOST})

  ExternalProject_Add(${proj}
    URL ${NIFTK_LOCATION_BOOST}
    URL_MD5 ${NIFTK_CHECKSUM_BOOST}
    CMAKE_GENERATOR ${GEN}
    CMAKE_ARGS
        ${EP_COMMON_ARGS}
        ${BOOST_ARGS}
        -DWITH_BZIP2:BOOL=OFF
        -DWITH_DOXYGEN:BOOL=OFF
        -DWITH_EXPAT:BOOL=OFF
        -DWITH_ICU:BOOL=OFF
        -DWITH_PYTHON:BOOL=OFF
        -DWITH_XSLTPROC:BOOL=OFF
        -DWITH_VALGRIND:BOOL=OFF
        -DWITH_ZLIB:BOOL=OFF
        -DCMAKE_INSTALL_PREFIX:PATH=${EP_BASE}/Install/${proj}
     DEPENDS ${proj_DEPENDENCIES}
    )
  set(BOOST_ROOT ${proj_INSTALL})
  if(WIN32)
    set(BOOST_INCLUDEDIR "${BOOST_ROOT}/include")
    set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib")
  else(WIN32)
    set(BOOST_INCLUDEDIR "${BOOST_ROOT}/include/boost-${NIFTK_VERSION_BOOST}")
    set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib/boost-${NIFTK_VERSION_BOOST}")
  endif(WIN32)

  message("SuperBuild loading Boost from ${BOOST_ROOT}")
  message("SuperBuild loading Boost using BOOST_INCLUDEDIR=${BOOST_INCLUDEDIR}")
  message("SuperBuild loading Boost using BOOST_LIBRARYDIR=${BOOST_LIBRARYDIR}")

else()

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif()
