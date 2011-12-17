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
#  Last Changed      : $LastChangedDate: 2011-05-25 18:13:39 +0100 (Wed, 25 May 2011) $ 
#  Revision          : $Revision: 6268 $
#  Last modified by  : $Author: kkl $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

#-----------------------------------------------------------------------------
# OPENIGTLINK
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED OpenIGTLink_DIR AND NOT EXISTS ${OpenIGTLink_DIR})
  MESSAGE(FATAL_ERROR "OpenIGTLink_DIR variable is defined but corresponds to non-existing directory \"${OpenIGTLink_DIR}\".")
ENDIF()

IF(BUILD_OPENIGTLINK)

  SET(proj OPENIGTLINK)
  SET(proj_DEPENDENCIES )
  SET(OPENIGTLINK_DEPENDS ${proj})
   
  IF(NOT DEFINED OpenIGTLink_DIR)
  
    ExternalProject_Add(${proj}
      URL http://cmic.cs.ucl.ac.uk/platform/dependencies/OpenIGTLink-7802.tar.gz
      BINARY_DIR ${proj}-build
      INSTALL_COMMAND ""
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
          ${EP_COMMON_ARGS}
          -DCMAKE_INSTALL_PREFIX:PATH=${EP_BASE}/Install/${proj}
       DEPENDS ${proj_DEPENDENCIES}
      )
  
    SET(OpenIGTLink_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)
    SET(OpenIGTLink_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeExternals/Source/OPENIGTLINK)
    
    MESSAGE("SuperBuild loading OpenIGTLink from ${OpenIGTLink_DIR}")
  
  ELSE(NOT DEFINED OpenIGTLink_DIR)
  
    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
  
  ENDIF(NOT DEFINED OpenIGTLink_DIR)

ENDIF(BUILD_OPENIGTLINK)