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
  set(proj_INSTALL ${CMAKE_BINARY_DIR}/${proj}-install)

  if(NOT DEFINED CGAL_DIR)    
    ######################################################################
    # Configure the CGAL Superbuild, to decide which plugins we want.
    ######################################################################

    niftkMacroGetChecksum(NIFTK_CHECKSUM_CGAL ${NIFTK_LOCATION_CGAL})

    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj}-src
      BINARY_DIR ${proj}-build
      PREFIX ${proj}-cmake
      INSTALL_DIR ${proj}-install
      URL ${NIFTK_LOCATION_CGAL}
      URL_MD5 ${NIFTK_CHECKSUM_CGAL}
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
        -DBOOST_ROOT:PATH=${BOOST_ROOT}
        -DBoost_NO_SYSTEM_PATHS:BOOL=TRUE
        -DBOOST_INCLUDEDIR:PATH=${BOOST_INCLUDEDIR}
        -DBOOST_LIBRARYDIR:PATH=${BOOST_LIBRARYDIR}
        -DBoost_USE_STATIC_LIBS:BOOL=${BUILD_SHARED}
        -DCGAL_CFG_NO_STL:BOOL=OFF
        -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED}
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      DEPENDS ${proj_DEPENDENCIES}
      )
    set(CGAL_DIR "${proj_INSTALL}/lib/CGAL")
    set(CGAL_INCLUDE_DIRS "${proj_INSTALL}/include") 

  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()

  message("SuperBuild loading CGAL from ${CGAL_DIR}")

endif(BUILD_MESHING)
