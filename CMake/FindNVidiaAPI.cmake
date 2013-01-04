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
  
SET(NVAPI_FOUND 0)

# Note: This is ONLY valid On Windows
IF(WIN32)

  SET(NVAPI_POSSIBLE_INCLUDE_PATHS 
    "C:/Program Files/nvapi/"
    "C:/Program Files/R304-developer/"
    "E:/build/nvapi/"
    "E:/build/R304-developer/"
    "E:/NifTK/nvapi/"
    "E:/NifTK/R304-developer/"
  )

  FIND_PATH(NVAPI_INCLUDE_DIR
    NAMES nvapi.h
    PATHS ${NVAPI_POSSIBLE_INCLUDE_PATHS}
  )

  IF("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
    SET(NVAPI_ARCH_DIR amd64)
    SET(NVAPI_LIBRARY_NAME nvapi64)
  ELSE()
    SET(NVAPI_ARCH_DIR x86)
    SET(NVAPI_LIBRARY_NAME nvapi)
  ENDIF()

  SET(NVAPI_POSSIBLE_LIBRARY_PATHS
    ${NVAPI_INCLUDE_DIR}/${NVAPI_ARCH_DIR}
  )

  FIND_LIBRARY(NVAPI_LIBRARY
    NAMES ${NVAPI_LIBRARY_NAME}
    HINTS ${NVAPI_POSSIBLE_LIBRARY_PATHS} 
  )

  IF(NVAPI_INCLUDE_DIR AND NVAPI_LIBRARY)
    SET(NVAPI_FOUND 1)
  ENDIF()

# Note: This is ONLY valid On Windows
ENDIF(WIN32)

