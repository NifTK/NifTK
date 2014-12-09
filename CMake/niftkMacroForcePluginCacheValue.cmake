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

macro(niftkMacroForcePluginCacheValue list_of_plugins plugin_value)
  foreach(plugin ${list_of_plugins})
    string(REPLACE ":" "\;" target_info ${plugin})
    set(target_info_list ${target_info})
    list(GET target_info_list 0 plugin_name)
    set(${PROJECT_NAME}_${plugin_name} ${plugin_value} CACHE BOOL "Build the ${plugin_name} Plugin. " FORCE)
  endforeach()
endmacro()
