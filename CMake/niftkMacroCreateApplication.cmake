#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#  
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2011-12-16 09:02:17 +0000 (Fri, 16 Dec 2011) $ 
#  Revision          : $Revision: 8038 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/
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
  
  MITK_USE_MODULE(niftkMitkExt)
  MITK_USE_MODULE(qtsingleapplication)

  INCLUDE_DIRECTORIES(${ALL_INCLUDE_DIRECTORIES})

  IF(APPLE)
    SET( OSX_ICON_FILES ${CMAKE_CURRENT_SOURCE_DIR}/icon.icns) 
    SET_SOURCE_FILES_PROPERTIES( ${OSX_ICON_FILES} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
  ENDIF(APPLE)

  SET(app_sources ${MY_APP_NAME}.cpp ${OSX_ICON_FILES} ${OSX_LOGO_FILES} )

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
  
  SET(TMP_MACOSX_BUNDLE_NAMES ${MACOSX_BUNDLE_NAMES})
  SET(MACOSX_BUNDLE_NAMES ${MY_APP_NAME})
  
  FunctionCreateBlueBerryApplication(
    NAME ${MY_APP_NAME}
    SOURCES ${app_sources}
    DESCRIPTION "${PROJECT_NAME} - ${MY_APP_NAME} Application"
    PLUGINS ${_include_plugins}
    EXCLUDE_PLUGINS ${_exclude_plugins}
    ${_app_options}
  )

  IF(APPLE)
    SET_TARGET_PROPERTIES( ${MY_APP_NAME} PROPERTIES 
      MACOSX_BUNDLE_EXECUTABLE_NAME "${MY_APP_NAME}"
      MACOSX_BUNDLE_GUI_IDENTIFIER "${MY_APP_NAME}"
      MACOSX_BUNDLE_BUNDLE_NAME "${MY_APP_NAME}"
      MACOSX_BUNDLE_LONG_VERSION_STRING "${NIFTK_VERSION_STRING}"
      MACOSX_BUNDLE_SHORT_VERSION_STRING "${NIFTK_VERSION_STRING}"
      MACOSX_BUNDLE_ICON_FILE "icon.icns"
      MACOSX_BUNDLE_COPYRIGHT "${NIFTK_COPYRIGHT}"
    )
  ENDIF(APPLE)

  IF(WIN32)
    TARGET_LINK_LIBRARIES(${MY_APP_NAME}
      optimized PocoFoundation debug PocoFoundationd
      optimized PocoUtil debug PocoUtild
      optimized PocoXml debug PocoXmld
      org_blueberry_osgi
      ${ALL_LIBRARIES}
      ${QT_QTCORE_LIBRARY}
      ${QT_QTMAIN_LIBRARY}
    )
  ELSE(WIN32)
    TARGET_LINK_LIBRARIES(${MY_APP_NAME}
      org_blueberry_osgi
      ${ALL_LIBRARIES}
    )
  ENDIF(WIN32)

  #############################################################################
  # Restore this MACOSX_BUNDLE_NAMES variable. See long-winded note above.
  #############################################################################
  SET(MACOSX_BUNDLE_NAMES ${TMP_MACOSX_BUNDLE_NAMES})
  
ENDMACRO(NIFTK_CREATE_APPLICATION)
