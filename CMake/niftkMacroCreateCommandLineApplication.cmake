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
# Note: This should be a one stop shop for creating a command line app
# for either the Slicer Execution Model type, or a "normal" command line app.
# The logic starts to get tricky when we either are building a GUI, or not
# and the process differs depending on platform. The main caveat here, is
# that with BUILD_GUI=OFF, and for Linux only, we still want to correctly
# build and packaage all available command line apps for use on a cluster box.
#
# The aim of this macro is therefore to contain all the right logic.
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
    
  endif()

endmacro()
