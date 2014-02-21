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
# RTK
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED RTK_DIR AND NOT EXISTS ${RTK_DIR})
  message(FATAL_ERROR "RTK_DIR variable is defined but corresponds to non-existing directory \"${RTK_DIR}\".")
endif()

if(BUILD_RTK)

  set(proj RTK)
  set(proj_DEPENDENCIES ITK)
  set(RTK_DEPENDS ${proj})

  if(NOT DEFINED RTK_DIR)

    set(additional_cmake_args )
    niftkMacroGetChecksum(NIFTK_CHECKSUM_RTK ${NIFTK_LOCATION_RTK})

    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj}-src
      BINARY_DIR ${proj}-build
      PREFIX ${proj}-cmake
      INSTALL_DIR ${proj}-install
      URL ${NIFTK_LOCATION_RTK}
      URL_MD5 ${NIFTK_CHECKSUM_RTK}
      INSTALL_COMMAND ""
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        ${additional_cmake_args}
        -DITK_DIR:PATH=${ITK_DIR}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(RTK_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
    message("SuperBuild loading RTK from ${RTK_DIR}")

  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()
endif()
