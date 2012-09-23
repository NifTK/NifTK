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
PROJECT(NiftyView)

MITK_USE_MODULE(niftkMitkExt)
MITK_USE_MODULE(qtsingleapplication)

INCLUDE_DIRECTORIES(${ALL_INCLUDE_DIRECTORIES})

IF(APPLE)
  SET( OSX_ICON_FILES ${CMAKE_CURRENT_SOURCE_DIR}/icon.icns) 
  SET_SOURCE_FILES_PROPERTIES( ${OSX_ICON_FILES} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
ENDIF(APPLE)

SET(app_sources ${MY_APP_NAME}.cpp ${OSX_ICON_FILES} ${OSX_LOGO_FILES} )

SET(_app_options)
IF(${MY_PROJECT_NAME}_SHOW_CONSOLE_WINDOW)
  LIST(APPEND _app_options SHOW_CONSOLE)
ENDIF()

SET(_exclude_plugins
)

FunctionCreateBlueBerryApplication(
  NAME ${MY_APP_NAME}
  SOURCES ${app_sources}
  DESCRIPTION "${PROJECT_NAME} - ${MY_APP_NAME} Application"
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
