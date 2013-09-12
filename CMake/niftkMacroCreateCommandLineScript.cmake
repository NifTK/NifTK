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

###############################################################################
#! \brief One stop shop for creating a command line script, with optional
#!        deployment of a batch/shell file for a Slicer Execution Model CLI.
#!
#! \param NAME <script name>  Specify the script name without file extension.
#! \param EXTENSION <ext>     Specify the scripts file extension, eg. py or sh
#! \param INSTALL_CLI_MODULES If specified, will generate a script that goes
#!                            into the cli-modules folder.
###############################################################################

macro(NIFTK_CREATE_COMMAND_LINE_SCRIPT)
  set(oneValueArgs NAME EXTENSION)
  set(options INSTALL_CLI_MODULES)
  set(multiValueArgs)
  cmake_parse_arguments(_APP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT _APP_NAME)
    message(FATAL_ERROR "NAME argument cannot be empty.")
  endif()

  # First, the script always goes in the bin folder.
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${_APP_NAME}.${_APP_EXTENSION}.in ${EXECUTABLE_OUTPUT_PATH}/${_APP_NAME}.${_APP_EXTENSION} @ONLY )
  MITK_INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/${_APP_NAME}.${_APP_EXTENSION})

  # In addition, if INSTALL_CLI_MODULES is specified, will generate a script into cli-modules folder.
  if(_APP_INSTALL_CLI_MODULES)

    message(STATUS "Configuring SEM CLI module from script: ${_APP_NAME}.${_APP_EXTENSION}")

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
  endif()
endmacro()
