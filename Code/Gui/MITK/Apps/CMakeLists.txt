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

# Note: The name of the App in Apps.cmake, must match the directory name.

foreach(NIFTK_APP ${NIFTK_APPS})
  
  # extract option_name
  string(REPLACE "^^" "\\;" target_info ${NIFTK_APP})
  set(target_info_list ${target_info})
  list(GET target_info_list 1 option_name)
  list(GET target_info_list 0 app_name)

  if(${option_name} OR NIFTK_BUILD_ALL_APPS)
    add_subdirectory(${app_name})
  endif()
      
endforeach()
