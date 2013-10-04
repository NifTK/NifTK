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
    message(FATAL_ERROR "NIFTK_CREATE_COMMAND_LINE_SCRIPT: NAME argument cannot be empty.")
  endif()

  set(FULL_APP_NAME "${_APP_NAME}.${_APP_EXTENSION}")

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FULL_APP_NAME}.in ${EXECUTABLE_OUTPUT_PATH}/${FULL_APP_NAME} @ONLY )
  MITK_INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/${FULL_APP_NAME})

  if(_APP_INSTALL_CLI_MODULES)
    message(STATUS "Configuring SEM CLI module from script: ${FULL_APP_NAME}")
    NIFTK_GENERATE_CLI_SCRIPT(NAME ${_APP_NAME})
  endif()
endmacro()
