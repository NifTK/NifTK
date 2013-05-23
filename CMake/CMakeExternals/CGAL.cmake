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
# CGAL
#-----------------------------------------------------------------------------

set(NIFTK_LOCATION_CGAL "http://cmic.cs.ucl.ac.uk/platform/dependencies/CGAL-3.8.tar.gz")

if(BUILD_MESHING)

  set(proj CGAL)
  set(proj_DEPENDENCIES BOOST)
  set(CGAL_DEPENDS ${proj})
  set(proj_INSTALL ${EP_BASE}/Install/${proj})

  if(NOT DEFINED CGAL_DIR)    
    ######################################################################
    # Configure the CGAL Superbuild, to decide which plugins we want.
    ######################################################################

    if (WIN32)
      set(BOOST_THREAD_LIB "${BOOST_LIBRARYDIR}/libboost_thread-vc90-mt-gd-1_46_1.lib")
      set(BUILD_SHARED OFF)
    else (WIN32)
      set(BOOST_THREAD_LIB "${BOOST_LIBRARYDIR}/libboost_thread-mt.a")
      set(BUILD_SHARED ON)
    endif (WIN32)

    niftkMacroGetChecksum(NIFTK_CHECKSUM_CGAL ${NIFTK_LOCATION_CGAL})

    ExternalProject_Add(${proj}
      URL ${NIFTK_LOCATION_CGAL}
      URL_MD5 ${NIFTK_CHECKSUM_CGAL}
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
        -DBOOST_ROOT:PATH=${BOOST_ROOT}
        -DBoost_INCLUDE_DIR:PATH=${BOOST_INCLUDEDIR}
        -DBoost_LIBRARY_DIRS:PATH=${BOOST_LIBRARYDIR}
        -DBoost_THREAD_LIBRARY:PATH=${BOOST_THREAD_LIB}
        -DBoost_THREAD_LIBRARY_DEBUG:PATH=${BOOST_THREAD_LIB}
        -DBoost_THREAD_LIBRARY_RELEASE:PATH=${BOOST_THREAD_LIB} 
        -DCGAL_CFG_NO_STL:BOOL=OFF
        -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED}
        -DCMAKE_INSTALL_PREFIX:PATH=${EP_BASE}/Install/${proj}
      DEPENDS ${proj_DEPENDENCIES}
      )
    set(CGAL_DIR "${EP_BASE}/Install/${proj}/lib/CGAL")
    set(CGAL_INCLUDE_DIRS "${EP_BASE}/Install/${proj}/include") 
  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()

  message("SuperBuild loading CGAL from ${CGAL_DIR}")

endif(BUILD_MESHING)
