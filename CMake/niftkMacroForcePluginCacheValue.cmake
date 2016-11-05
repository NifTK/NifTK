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

# This macro switches on the  plugins that are required by any of the applications
# and switches off the ones that are not needed by any of them.
#
# We need to forcibly overwrite the cache values so that if someone switches off
# an application at CMake configuration, all its required plugins are turned off
# and not built, unless they are required by another application.

macro(niftkMacroForcePluginCacheValues list_of_plugins)

  foreach (plugin ${list_of_plugins})
    string(REPLACE ":" "\;" target_info ${plugin})
    set(target_info_list ${target_info})
    list(GET target_info_list 0 plugin_name)
    list(GET target_info_list 1 plugin_value)

    if (plugin_value STREQUAL ON)
      set(${PROJECT_NAME}_${plugin_name} ON CACHE BOOL "Build the ${plugin_name} Plugin. " FORCE)
    elseif (plugin_value STREQUAL OFF)
      set(${PROJECT_NAME}_${plugin_name} OFF CACHE BOOL "Build the ${plugin_name} Plugin. " FORCE)
    endif()
  endforeach()

endmacro()
