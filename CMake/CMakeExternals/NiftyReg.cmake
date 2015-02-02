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
# NiftyReg
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED NIFTYREG_ROOT AND NOT EXISTS ${NIFTYREG_ROOT})
  message(FATAL_ERROR "NIFTYREG_ROOT variable is defined but corresponds to non-existing directory \"${NIFTYREG_ROOT}\".")
endif()

if(BUILD_NIFTYREG)

  niftkMacroDefineExternalProjectVariables(NiftyReg ${NIFTK_VERSION_NiftyReg})

  if(NOT DEFINED NIFTYREG_ROOT)

    niftkMacroGetChecksum(NIFTK_CHECKSUM_NiftyReg ${NIFTK_LOCATION_NiftyReg})

    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj_SOURCE}
      PREFIX ${proj_CONFIG}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${NIFTK_LOCATION_NiftyReg}
      URL_MD5 ${NIFTK_CHECKSUM_NiftyReg}
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_ALL_DEP:BOOL=ON
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DUSE_CUDA:BOOL=OFF
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(NIFTYREG_ROOT ${proj_INSTALL})

    message("SuperBuild loading NiftyReg from ${NIFTYREG_ROOT}")

  else(NOT DEFINED NIFTYREG_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED NIFTYREG_ROOT)

endif(BUILD_NIFTYREG)
