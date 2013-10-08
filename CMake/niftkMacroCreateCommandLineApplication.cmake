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
#! \brief One stop shop for creating a command line application.
#!
#! This function should cope with either Slicer Execution Model type or
#! 'normal' command line apps.
#!
#! \param NAME <app name> Specify the application name, where the application
#!                        name determines the name of the .cxx and .xml file.
#!                        So for MyApp, there should be MyApp.cxx and MyApp.xml
#!                        in the same folder.
#! \param BUILD_SLICER Configure the xml file, and calls SEMMacroBuildNifTKCLI
#!                     to generate a Slicer Execution Model type app.
#! \param BUILD_CLI    Build a normal command line app.
#!                     BUILD_SLICER and BUILD_CLI are mutually exclusive.
#! \param INSTALL_SCRIPT Will generate a script that goes into the cli-modules
#!                       folder.
###############################################################################

macro(NIFTK_CREATE_COMMAND_LINE_APPLICATION)
  set(options BUILD_SLICER BUILD_CLI INSTALL_SCRIPT)
  set(oneValueArgs NAME)
  set(multiValueArgs TARGET_LIBRARIES)
  cmake_parse_arguments(_APP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT _APP_NAME)
    message(FATAL_ERROR "NIFTK_CREATE_COMMAND_LINE_APPLICATION: NAME argument cannot be empty.")
  endif()

  set(FULL_APP_NAME ${_APP_NAME})

  if(_APP_BUILD_SLICER)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${_APP_NAME}.xml.in ${CMAKE_CURRENT_BINARY_DIR}/${_APP_NAME}.xml @ONLY )
    SEMMacroBuildNifTKCLI(
      NAME ${_APP_NAME}
      EXECUTABLE_ONLY
      ${CMAKE_CURRENT_BINARY_DIR}
      TARGET_LIBRARIES ${_APP_TARGET_LIBRARIES}
    )
  endif()

  if(_APP_BUILD_CLI)
    add_executable(${_APP_NAME} ${_APP_NAME}.cxx )
    target_link_libraries(${_APP_NAME} ${_APP_TARGET_LIBRARIES} )
    MITK_INSTALL(TARGETS ${_APP_NAME})
  endif()

  if(_APP_INSTALL_SCRIPT)
    NIFTK_GENERATE_CLI_SCRIPT(NAME ${_APP_NAME})
  endif()
endmacro()
