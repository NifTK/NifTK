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
# GDCM
#
# Note: NifTK as such doesn't need GDCM. However, if we use MITK,
# then MITK needs a version of ITK that has been built with a specfic 
# version of GDCM. So we build GDCM, and then ITK in that same fashion.
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED GDCM_DIR AND NOT EXISTS ${GDCM_DIR})
  MESSAGE(FATAL_ERROR "GDCM_DIR variable is defined but corresponds to non-existing directory \"${GDCM_DIR}\".")
ENDIF()

SET(proj GDCM)
SET(proj_DEPENDENCIES )
SET(GDCM_DEPENDS ${proj})

IF(NOT DEFINED GDCM_DIR)
  ExternalProject_Add(${proj}
     URL ${NIFTK_LOCATION_GDCM}
     BINARY_DIR ${proj}-build
     INSTALL_COMMAND ""
     PATCH_COMMAND ${CMAKE_COMMAND} -DTEMPLATE_FILE:FILEPATH=${CMAKE_SOURCE_DIR}/CMake/CMakeExternals/EmptyFileForPatching.dummy -P ${CMAKE_SOURCE_DIR}/CMake/CMakeExternals/PatchGDCM-2.0.18.cmake
     CMAKE_GENERATOR ${GEN}
     CMAKE_ARGS
       ${EP_COMMON_ARGS}
       -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
       -DGDCM_BUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
       -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
       -DBUILD_EXAMPLES:BOOL=${EP_BUILD_EXAMPLES}
     DEPENDS ${proj_DEPENDENCIES}
    )
  SET(GDCM_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)
  MESSAGE("SuperBuild loading GDCM from ${GDCM_DIR}")

  SET(GDCM_IS_2_0_18 TRUE)
  
ELSE()

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
  
  FIND_PACKAGE(GDCM)

  IF( GDCM_BUILD_VERSION EQUAL "18")
    SET(GDCM_IS_2_0_18 TRUE)
  ELSE()
    SET(GDCM_IS_2_0_18 FALSE)
  ENDIF()
 
ENDIF()
