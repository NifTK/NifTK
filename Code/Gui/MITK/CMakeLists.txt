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

####################################################################################
# Set the list of MITK based GUI application names. So, you can add more apps here.
# However, you will also need to specify stuff in each subdirectory.
####################################################################################

SET(MY_PROJECT_NAME NIFTK)
SET(MY_APP_NAME NiftyView)

OPTION(${MY_PROJECT_NAME}_BUILD_ALL_PLUGINS "Build all ${MY_PROJECT_NAME} plugins" OFF)
OPTION(${MY_PROJECT_NAME}_BUILD_ALL_APPS "Build all ${MY_PROJECT_NAME} applications" OFF)
OPTION(${MY_PROJECT_NAME}_SHOW_CONSOLE_WINDOW "Use this to enable or disable the console window when starting GUI Applications" ON)

MARK_AS_ADVANCED(${MY_PROJECT_NAME}_INSTALL_RPATH_RELATIVE
                 ${MY_PROJECT_NAME}_BUILD_ALL_PLUGINS
                 ${MY_PROJECT_NAME}_BUILD_ALL_APPS
                 ${MY_PROJECT_NAME}_SHOW_CONSOLE_WINDOW
                 )

IF(MITK_USE_QT)
  SET(QT_QMAKE_EXECUTABLE ${MITK_QMAKE_EXECUTABLE})
  ADD_DEFINITIONS(-DQWT_DLL)
ENDIF()

####################################################################################
# On Mac OSX all CTK plugins get copied into every application bundle.
####################################################################################

SET(MACOSX_BUNDLE_NAMES)
IF(APPLE)
  LIST(APPEND MACOSX_BUNDLE_NAMES ${MY_APP_NAME} )
ENDIF(APPLE)

####################################################################################
# Add 'Modules', i.e. shared libs
####################################################################################

# This project's directory holding module config files
set(${PROJECT_NAME}_MODULES_CONF_DIR "${PROJECT_BINARY_DIR}/${MODULES_CONF_DIRNAME}")

# Append this projects's module config directory to the global list
# (This is used to get include directories for the <module_name>Exports.h files right)
list(APPEND MODULES_CONF_DIRS ${${PROJECT_NAME}_MODULES_CONF_DIR})

# Clean the modulesConf directory. This ensures that modules are sorted
# according to their dependencies in the Modules/CMakeLists.txt file
file(GLOB _modules_conf_files ${${PROJECT_NAME}_MODULES_CONF_DIR}/*.cmake)
if(_modules_conf_files)
  file(REMOVE ${_modules_conf_files})
endif()

ADD_SUBDIRECTORY(Modules)

####################################################################################
# Create all CTK plugins.
####################################################################################

macro(GetMyTargetLibraries all_target_libraries varname)
  set(re_ctkplugin "^uk_ac_ucl_[a-zA-Z0-9_]+$")
  set(_tmp_list)
  list(APPEND _tmp_list ${all_target_libraries})
  ctkMacroListFilter(_tmp_list re_ctkplugin OUTPUT_VARIABLE ${varname})
endmacro()

# Get infos about application directories and build options
include("${CMAKE_CURRENT_SOURCE_DIR}/Apps/Apps.cmake")
set(_apps_fullpath )
foreach(_app ${PROJECT_APPS})
  list(APPEND _apps_fullpath "${CMAKE_CURRENT_SOURCE_DIR}/${_app}")
endforeach()

include(${CMAKE_CURRENT_SOURCE_DIR}/Plugins/Plugins.cmake)
ctkMacroSetupPlugins(${PROJECT_PLUGINS}
                     APPS ${_apps_fullpath} 
                     BUILD_OPTION_PREFIX ${PROJECT_NAME}_
                     BUILD_ALL ${${PROJECT_NAME}_BUILD_ALL_PLUGINS})

####################################################################################
# Add subdirectories, one for each application listed above.
####################################################################################

add_subdirectory(Apps)

