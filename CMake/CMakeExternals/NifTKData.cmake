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
# NifTKData - Downloads the unit-testing data as a separate project.
#-----------------------------------------------------------------------------

# Sanity checks
if (DEFINED NIFTK_DATA_DIR AND NOT EXISTS ${NIFTK_DATA_DIR})
  message(FATAL_ERROR "NIFTK_DATA_DIR variable is defined but corresponds to non-existing directory \"${NIFTK_DATA_DIR}\".")
endif ()

if (BUILD_TESTING)

  set(proj NifTKData)
  set(proj_DEPENDENCIES )
  set(NifTKData_DEPENDS ${proj})

  # Supported values: git, svn, tar
  set(${proj}_archtype ${NIFTK_ARCHTYPE_DATA})
  if (NOT DEFINED ${proj}_archtype)
    set(${proj}_archtype "git")
  endif()

  if (NOT DEFINED NIFTK_DATA_DIR)

    if (${proj}_archtype STREQUAL "git")
      set(${proj}_version ${NIFTK_VERSION_DATA_GIT})
      set(${proj}_location ${NIFTK_LOCATION_DATA_GIT})
      set(${proj}_location_options
        GIT_REPOSITORY ${${proj}_location}
        GIT_TAG ${${proj}_version}
      )
    elseif (${proj}_archtype STREQUAL "svn")
      set(${proj}_version ${NIFTK_VERSION_DATA_SVN})
      set(${proj}_location ${NIFTK_LOCATION_DATA_SVN})
      set(${proj}_location_options
        SVN_REPOSITORY ${${proj}_location}
        SVN_REVISION ${${proj}_version}
        SVN_TRUST_CERT 1
      )
    elseif (${proj}_archtype STREQUAL "tar")
      set(${proj}_version ${NIFTK_VERSION_DATA_TAR})
      set(${proj}_location ${NIFTK_LOCATION_DATA_TAR})
      niftkMacroGetChecksum(${proj}_checksum ${${proj}_location})
      set(${proj}_location_options
        URL ${${proj}_location}
        URL_MD5 ${${proj}_checksum}
      )
    else ()
      message("Unknown archive type. Valid values are git, svn and tar. Cannot download ${proj}.")
    endif ()

    ExternalProject_Add(${proj}
      ${${proj}_location_options}
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${NIFTK_VERSION_DATA_TAR}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      DEPENDS ${proj_DEPENDENCIES}
    )
    
    set(NIFTK_DATA_DIR ${EP_BASE}/Source/${proj})
    message("SuperBuild loading ${proj} from ${NIFTK_DATA_DIR}")
    
  else ()
  
    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
    
  endif (NOT DEFINED NIFTK_DATA_DIR)

endif (BUILD_TESTING)
