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
# NiftySeg
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED NIFTYSEG_ROOT AND NOT EXISTS ${NIFTYSEG_ROOT})
  message(FATAL_ERROR "NIFTYSEG_ROOT variable is defined but corresponds to non-existing disegtory \"${NIFTYSEG_ROOT}\".")
endif()

if(BUILD_NIFTYSEG)

  set(NIFTK_VERSION_NiftySeg "b2decf5160" CACHE STRING "Version of NiftySeg" FORCE)
  set(NIFTK_LOCATION_NiftySeg "${NIFTK_EP_TARBALL_LOCATION}/NiftySeg-${NIFTK_VERSION_NiftySeg}.tar.gz" CACHE STRING  "Location of NiftySeg" FORCE)

  niftkMacroDefineExternalProjectVariables(NiftySeg ${NIFTK_VERSION_NiftySeg})
  set(proj_DEPENDENCIES Eigen)

  if(NOT DEFINED NIFTYSEG_ROOT)

    niftkMacroGetChecksum(NIFTK_CHECKSUM_NiftySeg ${NIFTK_LOCATION_NiftySeg})

    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj_SOURCE}
      PREFIX ${proj_CONFIG}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${NIFTK_LOCATION_NiftySeg}
      URL_MD5 ${NIFTK_CHECKSUM_NiftySeg}
      CMAKE_GENERATOR ${GEN}
      #CONFIGURE_COMMAND ""
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${proj_VERSION}
      #BUILD_COMMAND ""
      #INSTALL_COMMAND ""
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DUSE_CUDA:BOOL=${NIFTK_USE_CUDA}
        -DUSE_OPENMP:BOOL=OFF
        -DINSTALL_PRIORS:BOOL=ON
        -DINSTALL_PRIORS_DIRECTORY:PATH=${proj_INSTALL}/priors
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
        -DUSE_SYSTEM_EIGEN=ON
        -DEigen_INCLUDE_DIR=${Eigen_INCLUDE_DIR}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(NIFTYSEG_ROOT ${proj_INSTALL})
    set(NIFTYSEG_INCLUDE_DIR "${NIFTYSEG_ROOT}/include")
    set(NIFTYSEG_LIBRARY_DIR "${NIFTYSEG_ROOT}/lib")

    message("SuperBuild loading NiftySeg from ${NIFTYSEG_ROOT}")

  else(NOT DEFINED NIFTYSEG_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED NIFTYSEG_ROOT)

endif(BUILD_NIFTYSEG)
