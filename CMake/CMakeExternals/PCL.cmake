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

  set(version "83c02003a2")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/PointCloudLibrary-pcl-${version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(PCL ${version} ${location})
  set(proj_DEPENDENCIES Boost Eigen FLANN VTK)

  if(NOT DEFINED PCL_DIR)

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
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
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DCMAKE_DEBUG_POSTFIX:STRING=
        -DBOOST_ROOT:PATH=${BOOST_ROOT}
        -DBOOST_INCLUDEDIR:PATH=${BOOST_ROOT}/include
        -DBOOST_LIBRARYDIR:PATH=${BOOST_ROOT}/lib
        -DBoost_NO_SYSTEM_PATHS:BOOL=ON
        -DEIGEN_ROOT:PATH=${Eigen_SOURCE_DIR}
        -DEIGEN_INCLUDE_DIR:PATH=${Eigen_SOURCE_DIR}
        -DFLANN_ROOT:PATH=${FLANN_DIR}
        -DVTK_DIR:PATH=${VTK_DIR}
        # explicitly define this (with the default value) because pcl will try to use static libs otherwise
        -DBoost_USE_STATIC_LIBS:BOOL=${Boost_USE_STATIC_LIBS}
        -DPCL_BUILD_WITH_BOOST_DYNAMIC_LINKING_WIN32:BOOL=NOT ${Boost_USE_STATIC_LIBS}
        -DBUILD_tools:BOOL=OFF
        -DBUILD_visualization:BOOL=OFF
      CMAKE_CACHE_ARGS
        ${EP_COMMON_CACHE_ARGS}
      CMAKE_CACHE_DEFAULT_ARGS
        ${EP_COMMON_CACHE_DEFAULT_ARGS}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(PCL_DIR ${proj_INSTALL})

    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})
    mitkFunctionInstallExternalCMakeProject(${proj})

    message("SuperBuild loading PCL from ${PCL_DIR}")

  else(NOT DEFINED PCL_DIR)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED PCL_DIR)

endif()
