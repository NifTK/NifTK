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
# NiftyLink
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED NIFTYLINK_DIR AND NOT EXISTS ${NIFTYLINK_DIR})
  MESSAGE(FATAL_ERROR "NIFTYLINK_DIR variable is defined but corresponds to non-existing directory \"${NIFTYLINK_DIR}\".")
ENDIF()

IF(BUILD_IGI)

  SET(proj NIFTYLINK)
  SET(proj_DEPENDENCIES)
  SET(NIFTYLINK_DEPENDS ${proj})
  
  IF(NOT DEFINED NIFTYLINK_DIR)
  
    ExternalProject_Add(${proj}
       GIT_REPOSITORY git:${NIFTK_LOCATION_NIFTYLINK}
       BINARY_DIR ${proj}-build
       INSTALL_COMMAND ""
       CMAKE_GENERATOR ${GEN}
       CMAKE_ARGS
         ${EP_COMMON_ARGS}
         -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
         -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
       DEPENDS ${proj_DEPENDENCIES}
    )
   
    SET(NIFTYLINK_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build/NiftyLink-build)
    SET(NIFTYLINK_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeExternals/Source/NIFTYLINK)
    SET(OpenIGTLink_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build/OPENIGTLINK-build)
    
    MESSAGE("SuperBuild loading NiftyLink from ${NIFTYLINK_DIR}")
    MESSAGE("SuperBuild loading OpenIGTLink from ${OpenIGTLink_DIR}")
    
  ELSE(NOT DEFINED NIFTYLINK_DIR)
  
    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
  
  ENDIF(NOT DEFINED NIFTYLINK_DIR)

ENDIF(BUILD_IGI)
