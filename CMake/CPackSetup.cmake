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


############################################################################
# First, set the generator variable
############################################################################

IF(NOT CPACK_GENERATOR)
  IF(WIN32)
  
    FIND_PROGRAM(NSIS_MAKENSIS NAMES makensis
      PATHS [HKEY_LOCAL_MACHINE\\SOFTWARE\\NSIS]
      DOC "Where is makensis.exe located"
      )

    IF(NOT NSIS_MAKENSIS)
      SET(CPACK_GENERATOR ZIP)
    ELSE()
      SET(CPACK_GENERATOR "NSIS")
    ENDIF(NOT NSIS_MAKENSIS)
    
    SET(CPACK_SOURCE_GENERATOR ZIP)
    
  ELSE()
  
    IF(APPLE)
      SET(CPACK_GENERATOR "DragNDrop")
    ELSE()
      SET(CPACK_GENERATOR TGZ)
    ENDIF()
    
    SET(CPACK_SOURCE_GENERATOR TGZ)
    
  ENDIF()
ENDIF(NOT CPACK_GENERATOR)

############################################################################
# This bit came from MITK (http://www.mitk.org). Don't know if we need it.
############################################################################
INCLUDE(InstallRequiredSystemLibraries)

############################################################################
# This bit came from Nifty Rec - START. Don't really know if we need it.
############################################################################
IF (CMAKE_SYSTEM_PROCESSOR MATCHES "unknown")
  SET (CMAKE_SYSTEM_PROCESSOR "x86")
ENDIF (CMAKE_SYSTEM_PROCESSOR MATCHES "unknown")
IF(NOT DEFINED CPACK_SYSTEM_NAME)
  SET(CPACK_SYSTEM_NAME ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})
ENDIF(NOT DEFINED CPACK_SYSTEM_NAME)
IF(${CPACK_SYSTEM_NAME} MATCHES Windows)
  IF(CMAKE_CL_64)
    SET(CPACK_SYSTEM_NAME Win64-${CMAKE_SYSTEM_PROCESSOR})
  ELSE(CMAKE_CL_64)
    SET(CPACK_SYSTEM_NAME Win32-${CMAKE_SYSTEM_PROCESSOR})
  ENDIF(CMAKE_CL_64)
ENDIF(${CPACK_SYSTEM_NAME} MATCHES Windows)

IF(${CPACK_SYSTEM_NAME} MATCHES Darwin AND CMAKE_OSX_ARCHITECTURES)
  list(LENGTH CMAKE_OSX_ARCHITECTURES _length)
  IF(_length GREATER 1)
    SET(CPACK_SYSTEM_NAME Darwin-Universal)
  ELSE(_length GREATER 1)
    SET(CPACK_SYSTEM_NAME Darwin-${CMAKE_OSX_ARCHITECTURES})
  ENDIF(_length GREATER 1)
ENDIF(${CPACK_SYSTEM_NAME} MATCHES Darwin AND CMAKE_OSX_ARCHITECTURES)

############################################################################
# This bit came from Nifty Rec - END
############################################################################

############################################################################
# The main setting of CPack settings that are independent of generator.
# See also CPackOptions.cmake.in, which gets modified at CMake time,
# and then used at CPack time.
############################################################################

SET(CPACK_PACKAGE_NAME "NifTK")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${CPACK_PACKAGE_NAME} - CMIC's translation medical imaging platform")
SET(CPACK_PACKAGE_VENDOR "Centre For Medical Image Computing (CMIC), University College London (UCL)")
SET(CPACK_PACKAGE_VERSION "${NIFTK_VERSION_MAJOR}.${NIFTK_VERSION_MINOR}")
SET(CPACK_PACKAGE_VERSION_MAJOR "${NIFTK_VERSION_MAJOR}")
SET(CPACK_PACKAGE_VERSION_MINOR "${NIFTK_VERSION_MINOR}")
SET(CPACK_CREATE_DESKTOP_LINKS "NiftyView")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_BINARY_DIR}/README.txt")
SET(CPACK_RESOURCE_FILE_README "${CMAKE_BINARY_DIR}/README.txt")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_BINARY_DIR}/LICENSE.txt")
SET(CPACK_RESOURCE_FILE_WELCOME "${CMAKE_BINARY_DIR}/INSTALLATION.txt")
SET(CPACK_PACKAGE_FILE_NAME "${NIFTK_DEPLOY_NAME}")
SET(CPACK_SOURCE_PACKAGE_FILE_NAME "${NIFTK_DEPLOY_NAME}")
SET(CPACK_MONOLITHIC_INSTALL ON)
