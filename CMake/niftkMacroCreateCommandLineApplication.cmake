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
#! \param INSTALL_LIBS If true will install all the dependent libraries.
#!                     If BUILD_GUI is on, then you can leave this off, as
#!                     there are only 3 GUI apps, but 120+ command line apps,
#!                     and if the GUIs run, and all the libraries have been
#!                     resolved and verified, then in all likelihood, so will
#!                     the command line apps. If however, BUILD_GUI is off,
#!                     you must specify this as true, and tolerate a long
#!                     long wait for 'make install' to complete.
#!
#!                     This ONLY works on LINUX.
#!
#!                     On Windows/Mac you are expected to distribute packages
#!                     with at least one GUI application. This parameter is
#!                     primarily just so we can create a cluster build of
#!                     all command line apps, with NO GUI.
###############################################################################

macro(NIFTK_CREATE_COMMAND_LINE_APPLICATION)
  set(options BUILD_SLICER BUILD_CLI INSTALL_SCRIPT )
  set(oneValueArgs NAME INSTALL_LIBS)
  set(multiValueArgs TARGET_LIBRARIES)
  cmake_parse_arguments(_APP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT _APP_NAME)
    message(FATAL_ERROR "NAME argument cannot be empty.")
  endif()

  set(MY_APP_NAME ${_APP_NAME})
  #message(STATUS "Creating command line application ${MY_APP_NAME}")

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

  ################
  # Only For Linux
  ################
  if(NOT WIN32 AND NOT APPLE)
    if(_APP_INSTALL_LIBS)

      # This part is based on that in niftkCreateGuiApplication.cmake.
      set(_library_dirs
        ${NiftyLink_LIBRARY_DIRS}
        ${curl_LIBRARY_DIR}
        ${zlib_LIBRARY_DIR}
        ${Boost_LIBRARY_DIRS}
      )
      if (${aruco_DIR})
        list(APPEND _library_dirs ${aruco_DIR}/lib)
      endif()


      # This part is based on that in mitkMacroInstallTargets.cmake
      set(intermediate_dir)
      if(WIN32 AND NOT MINGW)
        set(intermediate_dir ${CMAKE_BUILD_TYPE})
      endif()

      list(APPEND _library_dirs ${MITK_VTK_LIBRARY_DIRS}/${intermediate_dir})
      list(APPEND _library_dirs ${MITK_ITK_LIBRARY_DIRS}/${intermediate_dir})
      list(APPEND _library_dirs ${MITK_BINARY_DIR}/bin/${intermediate_dir})
      list(APPEND _library_dirs ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${intermediate_dir})

      if(_APP_BUILD_GUI)
        list(APPEND _library_dirs ${QT_LIBRARY_DIR})
        list(APPEND _library_dirs ${QT_LIBRARY_DIR}/../bin)
      endif()

      if(GDCM_DIR)
        list(APPEND _library_dirs ${GDCM_DIR}/bin/${intermediate_dir})
      endif()
      if(OpenCV_DIR)
        list(APPEND _library_dirs ${OpenCV_DIR}/bin/${intermediate_dir})
      endif()
      if(SOFA_DIR)
        list(APPEND _library_dirs ${SOFA_DIR}/bin/${intermediate_dir})
      endif()

      list(REMOVE_DUPLICATES _library_dirs)

      install(CODE "
        include(BundleUtilities)
        fixup_bundle(\"${CMAKE_INSTALL_PREFIX}/bin/${_APP_NAME}\"   \"\"   \"${_library_dirs}\")
        " COMPONENT Runtime)
    endif()
  endif()
endmacro()
