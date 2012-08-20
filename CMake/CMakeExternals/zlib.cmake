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
# zlib
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED zlib_DIR AND NOT EXISTS ${zlib_DIR})
  message(FATAL_ERROR "zlib_DIR variable is defined but corresponds to non-existing directory.")
endif()

set(proj zlib)
set(proj_DEPENDENCIES)
set(zlib_DEPENDS ${proj})

if(NOT DEFINED zlib_DIR)

  ExternalProject_Add(${proj}
    BINARY_DIR ${proj}-build
    URL http://zlib.net/zlib-1.2.7.tar.gz
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    CMAKE_GENERATOR ${GEN}
    CMAKE_ARGS
      ${EP_COMMON_ARGS}
      -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
      -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
    DEPENDS ${proj_DEPENDENCIES}
  )

  set(zlib_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build/zlib-build)
  set(zlib_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeExternals/Source/zlib)
  message("SuperBuild loading zlib from ${zlib_DIR}")

else(NOT DEFINED zlib_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED zlib_DIR)
