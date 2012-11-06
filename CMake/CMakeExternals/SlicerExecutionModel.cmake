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
# SlicerExecutionModel
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED SlicerExecutionModel_DIR AND NOT EXISTS ${SlicerExecutionModel_DIR})
  MESSAGE(FATAL_ERROR "SlicerExecutionModel_DIR variable is defined but corresponds to non-existing directory \"${SlicerExecutionModel_DIR}\".")
ENDIF()

SET(proj SlicerExecutionModel)
SET(proj_DEPENDENCIES ITK)
SET(SlicerExecutionModel_DEPENDS ${proj})

IF(NOT DEFINED SlicerExecutionModel_DIR)

#  niftkMacroGetChecksum(NIFTK_CHECKSUM_ITK ${NIFTK_LOCATION_ITK})

  ExternalProject_Add(${proj}
     GIT_REPOSITORY git://github.com/Slicer/SlicerExecutionModel.git
#     URL_MD5 ${NIFTK_CHECKSUM_ITK}
     BINARY_DIR ${proj}-build
     INSTALL_COMMAND ""
     CMAKE_GENERATOR ${GEN}
     CMAKE_ARGS
       ${EP_COMMON_ARGS}
       ${additional_cmake_args}
       -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
       -DBUILD_EXAMPLES:BOOL=${EP_BUILD_EXAMPLES}
       -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
       -DITK_DIR:PATH=${ITK_DIR}
     DEPENDS ${proj_DEPENDENCIES}
  )
 
  SET(SlicerExecutionModel_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)
  MESSAGE("SuperBuild loading SlicerExecutionModel from ${SlicerExecutionModel_DIR}")

ELSE(NOT DEFINED SlicerExecutionModel_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

ENDIF(NOT DEFINED SlicerExecutionModel_DIR)
