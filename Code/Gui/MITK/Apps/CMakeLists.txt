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
