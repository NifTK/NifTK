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
# PCL - Point Cloud Library..
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED PCL_DIR AND NOT EXISTS ${PCL_DIR})
  message(FATAL_ERROR "PCL_DIR variable is defined but corresponds to non-existing directory \"${PCL_DIR}\".")
endif()

if(BUILD_IGI AND BUILD_PCL)

  set(version "c2203fa60a")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/PointCloudLibrary-pcl-${version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(PCL ${version} ${location})
  set(proj_DEPENDENCIES Boost Eigen FLANN VTK)

  if(NOT DEFINED PCL_DIR)

    ExternalProject_Add(${proj}
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      UPDATE_COMMAND  ${GIT_EXECUTABLE} checkout ${proj_VERSION}
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DBOOST_ROOT:PATH=${BOOST_ROOT}
        -DBOOST_INCLUDEDIR:PATH=${BOOST_ROOT}/include
        -DBOOST_LIBRARYDIR:PATH=${BOOST_ROOT}/lib
        -DBoost_NO_SYSTEM_PATHS:BOOL=ON
        -DEIGEN_ROOT:PATH=${Eigen_DIR}
        -DFLANN_ROOT:PATH=${FLANN_DIR}
        -DVTK_DIR:PATH=${VTK_DIR}
        # explicitly define this (with the default value) because pcl will try to use static libs otherwise
        -DBoost_USE_STATIC_LIBS:BOOL=${Boost_USE_STATIC_LIBS}
        -DPCL_BUILD_WITH_BOOST_DYNAMIC_LINKING_WIN32:BOOL=NOT ${Boost_USE_STATIC_LIBS}
        -DBUILD_tools:BOOL=OFF
      DEPENDS ${proj_DEPENDENCIES}
    )
    if(WIN32)
      set(PCL_DIR ${proj_INSTALL}/cmake)
    else()
      set(PCL_DIR ${proj_INSTALL}/share/pcl-1.8)
    endif()
    message("SuperBuild loading PCL from ${PCL_DIR}")

  else(NOT DEFINED PCL_DIR)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED PCL_DIR)

endif()
