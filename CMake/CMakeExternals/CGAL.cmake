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

if(BUILD_MESHING)

  niftkMacroDefineExternalProjectVariables(CGAL ${NIFTK_VERSION_CGAL})
  set(proj_DEPENDENCIES Boost)

  if(NOT DEFINED CGAL_DIR)
    ######################################################################
    # Configure the CGAL Superbuild, to decide which plugins we want.
    ######################################################################

    niftkMacroGetChecksum(NIFTK_CHECKSUM_CGAL ${NIFTK_LOCATION_CGAL})

    if(UNIX)
      set(CGAL_CXX_FLAGS "${EP_COMMON_CXX_FLAGS} -fPIC")
    else()
      set(CGAL_CXX_FLAGS "${EP_COMMON_CXX_FLAGS}")
    endif(UNIX)

    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj_SOURCE}
      PREFIX ${proj_CONFIG}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${NIFTK_LOCATION_CGAL}
      URL_MD5 ${NIFTK_CHECKSUM_CGAL}
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
        -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
        -DBOOST_ROOT:PATH=${BOOST_ROOT}
        -DBoost_NO_SYSTEM_PATHS:BOOL=TRUE
        -DBOOST_INCLUDEDIR:PATH=${BOOST_INCLUDEDIR}
        -DBOOST_LIBRARYDIR:PATH=${BOOST_LIBRARYDIR}
        -DCGAL_CFG_NO_STL:BOOL=OFF
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
        -DWITH_OpenGL:BOOL=ON
        -DWITH_VTK:BOOL=ON
        -DVTK_DIR:PATH=${VTK_DIR}
        -DCMAKE_CXX_FLAGS:STRING=${CGAL_CXX_FLAGS}
      DEPENDS ${proj_DEPENDENCIES}
    )
    set(CGAL_DIR "${proj_INSTALL}/lib/CGAL")
    link_directories("${proj_INSTALL}/lib/")
  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()

  message("SuperBuild loading CGAL from ${CGAL_DIR}")

endif(BUILD_MESHING)
