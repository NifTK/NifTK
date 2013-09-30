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

####################################################################################
# Note: This is just to generate the wrapper script that goes in cli-modules folder.
####################################################################################

macro(NIFTK_GENERATE_CLI_SCRIPT)
  set(options)
  set(oneValueArgs NAME)
  set(multiValueArgs)
  cmake_parse_arguments(_APP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT _APP_NAME)
    message(FATAL_ERROR "NIFTK_GENERATE_CLI_SCRIPT: NAME argument cannot be empty.")
  endif()

  if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/CMake/CLI.bat.in ${EXECUTABLE_OUTPUT_PATH}/cli-modules/${_APP_NAME}.bat @ONLY )
    NIFTK_INSTALL_CLI_SCRIPT(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/cli-modules/${_APP_NAME}.bat)
  else(WIN32)
    if(APPLE)
      configure_file(${CMAKE_SOURCE_DIR}/CMake/CLI-For-Mac.sh.in ${EXECUTABLE_OUTPUT_PATH}/cli-modules/${_APP_NAME}.sh @ONLY )
      NIFTK_INSTALL_CLI_SCRIPT(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/cli-modules/${_APP_NAME}.sh)
    else()
      configure_file(${CMAKE_SOURCE_DIR}/CMake/CLI.sh.in ${EXECUTABLE_OUTPUT_PATH}/cli-modules/${_APP_NAME}.sh @ONLY )
      NIFTK_INSTALL_CLI_SCRIPT(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/cli-modules/${_APP_NAME}.sh)
    endif()
  endif(WIN32)
endmacro()
