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

  set(version "4.4-patched")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/CGAL-${version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(CGAL ${version} ${location})
  set(proj_DEPENDENCIES Boost)

  if(NOT DEFINED CGAL_DIR)
    ######################################################################
    # Configure the CGAL Superbuild, to decide which plugins we want.
    ######################################################################

    if(UNIX)
      set(CGAL_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    else()
      set(CGAL_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    endif(UNIX)

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DBOOST_ROOT:PATH=${BOOST_ROOT}
        -DBoost_NO_SYSTEM_PATHS:BOOL=TRUE
        -DBOOST_INCLUDEDIR:PATH=${BOOST_INCLUDEDIR}
        -DBOOST_LIBRARYDIR:PATH=${BOOST_LIBRARYDIR}
        -DCGAL_CFG_NO_STL:BOOL=OFF
        -DWITH_OpenGL:BOOL=ON
        -DWITH_VTK:BOOL=ON
        -DVTK_DIR:PATH=${VTK_DIR}
        -DCMAKE_CXX_FLAGS:STRING=${CGAL_CXX_FLAGS}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(CGAL_DIR "${proj_INSTALL}")
    link_directories("${proj_INSTALL}/lib")

    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})

  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()

  message("SuperBuild loading CGAL from ${CGAL_DIR}")

endif(BUILD_MESHING)
