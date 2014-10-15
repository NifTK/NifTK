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
# VL
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED VL_ROOT AND NOT EXISTS ${VL_ROOT})
  message(FATAL_ERROR "VL_ROOT variable is defined but corresponds to non-existing directory \"${VL_ROOT}\".")
endif()

if(BUILD_VL)

  set(proj VL)
  set(proj_DEPENDENCIES)
  set(proj_INSTALL ${CMAKE_BINARY_DIR}/${proj}-install )
  set(VL_DEPENDS ${proj})


  if(NOT DEFINED VL_DIR)

    set(revision_tag dev)
    
	if (NIFTK_VL_DEV)
      set(VL_location_options
        GIT_REPOSITORY ${NIFTK_LOCATION_VL_REPOSITORY}
        GIT_TAG ${revision_tag}
      )
    else ()
      # Must Not Leave Tarballs on Web Server
      # niftkMacroGetChecksum(NIFTK_CHECKSUM_NIFTYLINK ${NIFTK_LOCATION_VL_TARBALL})
      # set(VL_location_options
      #   URL ${NIFTK_LOCATION_VL_TARBALL}
      #   URL_MD5 ${NIFTK_CHECKSUM_VL}
      # )
      #
      # But we still want a specific version
      set(VL_location_options
        GIT_REPOSITORY ${NIFTK_LOCATION_VL_REPOSITORY}
        GIT_TAG ${NIFTK_VERSION_VL}
      )
    endif ()

    set(additional_cmake_args )

    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj}-src
      BINARY_DIR ${proj}-build
      PREFIX ${proj}-cmake
      INSTALL_DIR ${proj}-install
      ${VL_location_options}
      UPDATE_COMMAND  ${GIT_EXECUTABLE} checkout ${NIFTK_VERSION_VL}
      #INSTALL_COMMAND ""
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
      ${EP_COMMON_ARGS}
      ${additional_cmake_args}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
      -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
      -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      -DVL_GUI_QT4_SUPPORT:BOOL=${QT_FOUND}
      DEPENDS ${proj_DEPENDENCIES}
    )

	set(VL_ROOT ${proj_INSTALL})

    message("SuperBuild loading VL from ${VL_ROOT}")

  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()
endif()
