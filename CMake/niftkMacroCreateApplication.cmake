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

MACRO(NIFTK_CREATE_APPLICATION)
  MACRO_PARSE_ARGUMENTS(_APP
                        "NAME;INCLUDE_PLUGINS;EXCLUDE_PLUGINS"
                        ""
                        ${ARGN}
                        )

  IF(NOT _APP_NAME)
    MESSAGE(FATAL_ERROR "NAME argument cannot be empty.")
  ENDIF()
                        
  SET(MY_APP_NAME ${_APP_NAME})

  # The MITK_USE_MODULE sets up the include path for compile time...
  MITK_USE_MODULE(niftkCore)
  MITK_USE_MODULE(qtsingleapplication)
  INCLUDE_DIRECTORIES(${ALL_INCLUDE_DIRECTORIES})
  
  # ... and here we are specifying additional link time dependencies.
  SET(_link_libraries
    niftkCore
    qtsingleapplication
  )

  SET(_app_options)
  IF(${NIFTK_SHOW_CONSOLE_WINDOW})
    LIST(APPEND _app_options SHOW_CONSOLE)
  ENDIF()

  SET(_include_plugins
    ${_APP_INCLUDE_PLUGINS}
  )
  SET(_exclude_plugins
    ${_APP_EXCLUDE_PLUGINS}
  )
  
  # NOTE: Check CMake/PackageDepends for any additional dependencies.
  SET(_library_dirs
    ${NiftyLink_LIBRARY_DIRS}
    ${curl_LIBRARY_DIR}
    ${Boost_LIBRARY_DIRS}
    ${zlib_LIBRARY_DIR}
    ${aruco_DIR}/lib
  )
  
  #############################################################################
  # Watch out for this:
  # In the top level CMakeLists, MACOSX_BUNDLE_NAMES will contain all the apps
  # that we have available in this project. This is so that when you create a
  # Module, or a Plugin, under the hood, you are using an MITK macro, which 
  # takes care of copying said Module, Plugin into ALL bundles. HOWEVER, within
  # the FunctionCreateBlueBerryApplication, this has the side effect of
  # copying all MITK and CTK plugins into ALL bundles. This breaks the build at
  # install time, as you package up the first application and then when the 
  # second application is created, this function will again copy all the MITK 
  # and CTK plugins back into the first app, which overwrites all the library
  # @executable path settings leading to an invalid bundle for all but the 
  # last executable. So, we save the variable here, and restore it at the end
  # of this macro.
  #############################################################################
  
  IF(APPLE)
    SET(TMP_MACOSX_BUNDLE_NAMES ${MACOSX_BUNDLE_NAMES})
    SET(MACOSX_BUNDLE_NAMES ${MY_APP_NAME})
  ENDIF()
  
  FunctionCreateBlueBerryApplication(
    NAME ${MY_APP_NAME}
    SOURCES ${MY_APP_NAME}.cxx
    PLUGINS ${_include_plugins}
    EXCLUDE_PLUGINS ${_exclude_plugins}
    LINK_LIBRARIES ${_link_libraries}
    LIBRARY_DIRS ${_library_dirs}
    ${_app_options}
  )

  #############################################################################
  # Restore this MACOSX_BUNDLE_NAMES variable. See long-winded note above.
  #############################################################################
  IF(APPLE)
    SET(MACOSX_BUNDLE_NAMES ${TMP_MACOSX_BUNDLE_NAMES})
  ENDIF()
  
ENDMACRO(NIFTK_CREATE_APPLICATION)
