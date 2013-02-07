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

MACRO(NIFTK_CONFIGURE_AND_BUILD_CLI)
  MACRO_PARSE_ARGUMENTS(_APP
                        "NAME;TARGET_LIBRARIES"
                        ""
                        ${ARGN}
                        )

  IF(NOT _APP_NAME)
    MESSAGE(FATAL_ERROR "NAME argument cannot be empty.")
  ENDIF()

  CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/${_APP_NAME}.xml.in ${CMAKE_CURRENT_BINARY_DIR}/${_APP_NAME}.xml @ONLY )
    
  IF(WIN32) 
    CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/CLI.bat.in ${EXECUTABLE_OUTPUT_PATH}/cli-modules/${_APP_NAME}.bat @ONLY )
    NIFTK_INSTALL_CLI(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/cli-modules/${_APP_NAME}.bat)
  ELSE(WIN32)
    CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/CLI.sh.in ${EXECUTABLE_OUTPUT_PATH}/cli-modules/${_APP_NAME}.sh @ONLY )
    NIFTK_INSTALL_CLI(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/cli-modules/${_APP_NAME}.sh)
  ENDIF(WIN32)

  SEMMacroBuildNifTKCLI(
    NAME ${_APP_NAME}
    EXECUTABLE_ONLY
    ${CMAKE_CURRENT_BINARY_DIR}
    TARGET_LIBRARIES ${_APP_TARGET_LIBRARIES}
  )

ENDMACRO()
  