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
# NiftyRec
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED NIFTYREC_ROOT AND NOT EXISTS ${NIFTYREC_ROOT})
  message(FATAL_ERROR "NIFTYREC_ROOT variable is defined but corresponds to non-existing directory \"${NIFTYREC_ROOT}\".")
endif()

if(BUILD_NIFTYREC)

  set(proj NiftyRec)
  set(proj_DEPENDENCIES NiftyReg )
  set(proj_INSTALL ${CMAKE_BINARY_DIR}/${proj}-install )
  set(NIFTYREC_DEPENDS ${proj})

  if(NOT DEFINED NIFTYREC_ROOT)

    niftkMacroGetChecksum(NIFTK_CHECKSUM_NIFTYREC ${NIFTK_LOCATION_NIFTYREC})

    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj}-src
      BINARY_DIR ${proj}-build
      PREFIX ${proj}-cmake
      INSTALL_DIR ${proj}-install
      URL ${NIFTK_LOCATION_NIFTYREC}
      URL_MD5 ${NIFTK_CHECKSUM_NIFTYREC}
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
        -DUSE_CUDA:BOOL=${NIFTK_USE_CUDA}
        -DCUDA_SDK_ROOT_DIR=${CUDA_SDK_ROOT_DIR}
      DEPENDS ${proj_DEPENDENCIES}
      )

    set(NIFTYREC_ROOT ${proj_INSTALL})
    set(NIFTYREC_INCLUDE_DIR "${NIFTYREC_ROOT}/include")
    set(NIFTYREC_LIBRARY_DIR "${NIFTYREC_ROOT}/lib")

    message("SuperBuild loading NIFTYREC from ${NIFTYREC_ROOT}")

  else(NOT DEFINED NIFTYREC_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED NIFTYREC_ROOT)

endif(BUILD_NIFTYREC)
