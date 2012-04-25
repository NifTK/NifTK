#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2011-12-17 14:35:07 +0000 (Sat, 17 Dec 2011) $ 
#  Revision          : $Revision: 8065 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : stian.johnsen.09@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

#-----------------------------------------------------------------------------
# CGAL
#-----------------------------------------------------------------------------

IF(BUILD_MESHING)

  SET(proj CGAL)
  SET(proj_DEPENDENCIES BOOST)
  SET(CGAL_DEPENDS ${proj})
  SET(proj_INSTALL ${EP_BASE}/Install/${proj})

  IF(NOT DEFINED CGAL_DIR)    
    ######################################################################
    # Configure the CGAL Superbuild, to decide which plugins we want.
    ######################################################################

    IF (WIN32)
      SET(BOOST_THREAD_LIB "${BOOST_LIBRARYDIR}/libboost_thread-vc90-mt-gd-1_46_1.lib")
      SET(BUILD_SHARED OFF)
    ELSE (WIN32)
      SET(BOOST_THREAD_LIB "${BOOST_LIBRARYDIR}/libboost_thread-mt.a")
      SET(BUILD_SHARED ON)
    ENDIF (WIN32)

    ExternalProject_Add(${proj}
      URL http://cmic.cs.ucl.ac.uk/platform/dependencies/CGAL-3.8.tar.gz
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
    SET(CGAL_DIR "${EP_BASE}/Install/${proj}/lib/CGAL")
    SET(CGAL_INCLUDE_DIRS "${EP_BASE}/Install/${proj}/include") 
  ELSE()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  ENDIF()

  MESSAGE("SuperBuild loading CGAL from ${CGAL_DIR}")
  
ENDIF(BUILD_MESHING)
