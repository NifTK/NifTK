#/*================================================================================
#
#  NiftyGuide: A software package for light-weight client applications for 
#              Image Guided Interventions, developed at University College London.
#  
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2011-12-01 15:19:31 +0000 (Thu, 01 Dec 2011) $ 
#  Revision          : $Revision: 7901 $
#  Last modified by  : $Author: gerge $
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

SET(proj NIFTYLINK)
SET(proj_DEPENDENCIES OPENIGTLINK)
SET(NIFTYLINK_DEPENDS ${proj})

IF(NOT DEFINED NIFTYLINK_DIR)

  ExternalProject_Add(${proj}
     SVN_REPOSITORY https://cmicdev.cs.ucl.ac.uk/svn/cmic/trunk/NiftyLink
     SVN_REVISION -r 8148
     BINARY_DIR ${proj}-build
     INSTALL_COMMAND ""
     CMAKE_GENERATOR ${GEN}
     CMAKE_ARGS
       ${EP_COMMON_ARGS}
       -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
       -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
       -DOpenIGTLink_DIR:PATH=${OpenIGTLink_DIR}
     DEPENDS ${proj_DEPENDENCIES}
  )
 
  SET(NIFTYLINK_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build/NiftyLink-build)
  SET(NIFTYLINK_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeExternals/Source/NIFTYLINK)
  MESSAGE("SuperBuild loading NiftyLink from ${NIFTYLINK_DIR}")

ELSE(NOT DEFINED NIFTYLINK_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

ENDIF(NOT DEFINED NIFTYLINK_DIR)
