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

FOREACH(NIFTK_APP ${NIFTK_APPS})
  
  # extract option_name
  STRING(REPLACE "^^" "\\;" target_info ${NIFTK_APP})
  SET(target_info_list ${target_info})
  LIST(GET target_info_list 1 option_name)
  LIST(GET target_info_list 0 app_name)

  IF(${option_name} OR NIFTK_BUILD_ALL_APPS)
    ADD_SUBDIRECTORY(${app_name})
  ENDIF()
      
ENDFOREACH()
