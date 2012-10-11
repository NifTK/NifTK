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
IF(DEFINED NiftyLink_DIR AND NOT EXISTS ${NiftyLink_DIR})
  MESSAGE(FATAL_ERROR "NiftyLink_DIR variable is defined but corresponds to non-existing directory \"${NIFTYLINK_DIR}\".")
ENDIF()

IF(BUILD_IGI)

  SET(proj NiftyLink)
  SET(proj_DEPENDENCIES)
  SET(NIFTYLINK_DEPENDS ${proj})
  
  IF(NOT DEFINED NiftyLink_DIR)
  
    SET(revision_tag "development")

    IF(${proj}_REVISION_TAG)
      SET(revision_tag ${${proj}_REVISION_TAG})
    ENDIF()

    MESSAGE("Pulling NiftyLink from ${NIFTK_LOCATION_NIFTYLINK}")
    
    ExternalProject_Add(${proj}
       GIT_REPOSITORY ${NIFTK_LOCATION_NIFTYLINK}
       GIT_TAG ${revision_tag}
       BINARY_DIR ${proj}-build
       INSTALL_COMMAND ""
       CMAKE_GENERATOR ${GEN}
       CMAKE_ARGS
         ${EP_COMMON_ARGS}
         -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
         -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
         -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
       DEPENDS ${proj_DEPENDENCIES}
    )
   
    SET(NiftyLink_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build/NiftyLink-build)
    SET(NiftyLink_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeExternals/Source/NiftyLink)
    SET(OpenIGTLink_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build/OPENIGTLINK-build)
    
    MESSAGE("SuperBuild loading NiftyLink from ${NiftyLink_DIR}")
    MESSAGE("SuperBuild loading OpenIGTLink from ${OpenIGTLink_DIR}")
    
  ELSE(NOT DEFINED NiftyLink_DIR)
  
    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
  
  ENDIF(NOT DEFINED NiftyLink_DIR)

ENDIF(BUILD_IGI)
