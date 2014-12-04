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

####################################################################################
# Set the list of MITK based GUI application names. So, you can add more apps here.
# However, you will also need to specify stuff in each subdirectory.
####################################################################################

set(MY_PROJECT_NAME ${PROJECT_NAME})

mark_as_advanced(${MY_PROJECT_NAME}_INSTALL_RPATH_RELATIVE)

if(MITK_USE_QT)
  set(QT_QMAKE_EXECUTABLE ${MITK_QMAKE_EXECUTABLE})
  add_definitions(-DQWT_DLL)
endif()

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

add_subdirectory(Modules)

####################################################################################
# Create all CTK plugins.
####################################################################################

ctkFunctionGetAllPluginTargets(all_mitk_ctk_plugins)

macro(GetMyTargetLibraries all_target_libraries varname)
  set(re_ctkplugin "^(uk_ac_ucl|it_unito_cim)_[a-zA-Z0-9_]+$")
  set(_tmp_list)
  list(APPEND _tmp_list ${all_target_libraries})
  ctkMacroListFilter(_tmp_list re_ctkplugin OUTPUT_VARIABLE ${varname})
endmacro()

macro(ForcePluginCacheValue list_of_plugins plugin_value)
  foreach(plugin ${list_of_plugins})
    string(REPLACE ":" "\;" target_info ${plugin})
    set(target_info_list ${target_info})
    list(GET target_info_list 0 plugin_name)
    set(${PROJECT_NAME}_${plugin_name} ${plugin_value} CACHE BOOL "Build the ${plugin_name} Plugin. " FORCE)
  endforeach()
endmacro()

# Get infos about application directories and build options
include("${CMAKE_CURRENT_SOURCE_DIR}/Apps/Apps.cmake")
set(_apps_fullpath )
foreach(NIFTK_APP ${NIFTK_APPS})
  string(REPLACE "^^" "\\;" target_info ${NIFTK_APP})
  set(target_info_list ${target_info})
  list(GET target_info_list 1 option_name)
  list(GET target_info_list 0 app_name)
  if(${option_name} OR NIFTK_BUILD_ALL_APPS)
    list(APPEND _apps_fullpath "${CMAKE_CURRENT_SOURCE_DIR}/Apps/${NIFTK_APP}")
  endif()
endforeach()

# This is to get a list of plugins to pass to ctkMacroSetupPlugins
include(${CMAKE_CURRENT_SOURCE_DIR}/Plugins/Plugins.cmake)

# These are required, because people turn BUILD_IGI/BUILD_MIDAS off
# and expect all the corresponding plugins to do the same.
ForcePluginCacheValue(${IGI_PLUGINS} ${BUILD_IGI})
ForcePluginCacheValue(${MIDAS_PLUGINS} ${BUILD_MIDAS})

ctkMacroSetupPlugins(${PROJECT_PLUGINS}
                     APPS ${_apps_fullpath} 
                     BUILD_OPTION_PREFIX ${PROJECT_NAME}_
                     BUILD_ALL ${${PROJECT_NAME}_BUILD_ALL_PLUGINS})

####################################################################################
# Add subdirectories. 
#   Apps contains full MITK apps using MITK application framework BlueBerry.
#   HelperApps contains independent stuff like utilities, command line apps etc.
####################################################################################

if(BUILD_COMMAND_LINE_PROGRAMS)
  add_subdirectory(HelperApps)
endif()

add_subdirectory(Apps)

