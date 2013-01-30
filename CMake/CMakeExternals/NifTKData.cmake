#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2011-12-17 14:35:07 +0000 (Sat, 17 Dec 2011) $ 
#  Revision          : $Revision: 8065 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

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

  # Valid values: git, svn, tar
  set(${proj}_archtype "git")
  set(${proj}_version "d9fe07385c")
  set(${proj}_location "https://cmicdev.cs.ucl.ac.uk/git/NifTKData")

  #set(${proj}_archtype "svn")
  #set(${proj}_version "9749")
  #set(${proj}_location "https://cmicdev.cs.ucl.ac.uk/svn/cmic/trunk/NifTKData")

  #set(${proj}_archtype "tar")
  #set(${proj}_version "d9fe07385c")
  #set(${proj}_location "http://cmicdev.cs.ucl.ac.uk/platform/dependencies/${proj}-${${proj}_version}.tar.gz")

  if (NOT DEFINED NIFTK_DATA_DIR)

    if (${proj}_archtype STREQUAL "git")
      set(${proj}_location_options
        GIT_REPOSITORY ${${proj}_location}
        GIT_TAG ${${proj}_version}
      )
    elseif (${proj}_archtype STREQUAL "svn")
      set(${proj}_location_options
        SVN_REPOSITORY ${${proj}_location}
        SVN_REVISION ${${proj}_version}
        SVN_TRUST_CERT 1
      )
    elseif (${proj}_archtype STREQUAL "tar")
      niftkMacroGetChecksum(${proj}_checksum ${${proj}_location})
      set(${proj}_location_options
        URL ${${proj}_location}
        URL_MD5 ${${proj}_checksum}
      )
    else ()
      message("Unknown archtype. Cannot download ${proj}.")
    endif ()

    ExternalProject_Add(${proj}
      ${${proj}_location_options}
      UPDATE_COMMAND ""
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
