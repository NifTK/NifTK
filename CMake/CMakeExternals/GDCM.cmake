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
# GDCM
#
# Note: NifTK as such doesn't need GDCM. However, if we use MITK,
# then MITK needs a version of ITK that has been built with a specfic 
# version of GDCM. So we build GDCM, and then ITK in that same fashion.
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED GDCM_DIR AND NOT EXISTS ${GDCM_DIR})
  message(FATAL_ERROR "GDCM_DIR variable is defined but corresponds to non-existing directory \"${GDCM_DIR}\".")
endif()

# Check if an external ITK build tree was specified.
# If yes, use the GDCM from ITK, otherwise ITK will complain
if(ITK_DIR)
  find_package(ITK)
  if(ITK_GDCM_DIR)
    set(GDCM_DIR ${ITK_GDCM_DIR})
  endif()
endif()


set(proj GDCM)
set(proj_DEPENDENCIES )
set(GDCM_DEPENDS ${proj})

if(NOT DEFINED GDCM_DIR)

  niftkMacroGetChecksum(NIFTK_CHECKSUM_GDCM ${NIFTK_LOCATION_GDCM})

  ExternalProject_Add(${proj}
     URL ${NIFTK_LOCATION_GDCM}
     URL_MD5 ${NIFTK_CHECKSUM_GDCM}
     BINARY_DIR ${proj}-build
     INSTALL_COMMAND ""
     CMAKE_GENERATOR ${GEN}
     CMAKE_ARGS
       ${EP_COMMON_ARGS}
       -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
       -DGDCM_BUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
       -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
       -DBUILD_EXAMPLES:BOOL=${EP_BUILD_EXAMPLES}
     DEPENDS ${proj_DEPENDENCIES}
    )
  set(GDCM_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)
  message("SuperBuild loading GDCM from ${GDCM_DIR}")

else()

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  find_package(GDCM)

endif()
