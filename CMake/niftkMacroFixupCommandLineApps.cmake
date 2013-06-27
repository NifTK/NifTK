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
#! \brief For making sure all the necessary libraries are installed, for the
#         case when we are building command line apps, but no GUI on LINUX.
#!
#! \param APPS List of app names.
###############################################################################

macro(NIFTK_FIXUP_COMMAND_LINE_APPLICATIONS)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs APPS)
  cmake_parse_arguments(_install "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  ################
  # Only For Linux
  ################
  if(NOT WIN32 AND NOT APPLE)

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


    # This is slow, we have to validate apps one at a time.
    foreach(APP ${_install_APPS})
      set(_full_app_path "${CMAKE_INSTALL_PREFIX}/bin/${APP}")

      install(CODE "
        include(BundleUtilities)
        fixup_bundle(\"${_full_app_path}\"   \"\"   \"${_library_dirs}\")
        " COMPONENT Runtime)

    endforeach()

  endif()
endmacro()
