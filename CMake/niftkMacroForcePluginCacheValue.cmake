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

  set(_plugins_on )
  set(_plugins_off )

  foreach (plugin ${list_of_plugins})
    string(REPLACE ":" "\;" target_info ${plugin})
    set(target_info_list ${target_info})
    list(GET target_info_list 0 plugin_name)
    list(GET target_info_list 1 plugin_value)

    if (plugin_value STREQUAL ON)
      if (NOT plugin_name IN_LIST _plugins_on)
        list(APPEND _plugins_on ${plugin_name})
      endif()
      if (plugin_name IN_LIST _plugins_off)
        list(REMOVE_ITEM _plugins_off ${plugin_name})
      endif()
    elseif (plugin_value STREQUAL OFF)
      if (NOT plugin_name IN_LIST _plugins_off AND NOT plugin_name IN_LIST _plugins_off)
        list(APPEND _plugins_off ${plugin_name})
      endif()
    endif()
  endforeach()

  foreach (plugin ${_plugins_on})
    set(${PROJECT_NAME}_${plugin} ON CACHE BOOL "Build the ${plugin_name} Plugin. " FORCE)
  endforeach()

  foreach (plugin ${_plugins_off})
    set(${PROJECT_NAME}_${plugin} OFF CACHE BOOL "Build the ${plugin_name} Plugin. " FORCE)
  endforeach()

endmacro()
