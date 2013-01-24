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
#  Last Changed      : $LastChangedDate: 2011-12-16 15:42:03 +0000 (Fri, 16 Dec 2011) $ 
#  Revision          : $Revision: 8057 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

MACRO(NIFTK_ADD_MODULE module_name)
  INCLUDE(${${PROJECT_NAME}_MODULES_CONF_DIR}/${module_name}Config.cmake)
  INCLUDE_DIRECTORIES(${${module_name}_INCLUDE_DIRS})
  LINK_DIRECTORIES(${${module_name}_LIBRARY_DIRS})
  SET(HELPER_APP_LIBRARIES
    ${HELPER_APP_LIBRARIES}
    ${${module_name}_PROVIDES}
    ${${module_name}_DEPENDS}    
  )
ENDMACRO()

LINK_DIRECTORIES(${MITK_LIBRARY_DIRS})
INCLUDE_DIRECTORIES(${MITK_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${${PROJECT_NAME}_MODULES_CONF_DIR})

SET(HELPER_APP_LIBRARIES
  ${ALL_LIBRARIES}
)

SET(HELPER_APPS
  niftkCameraCalibration
)

SET(HELPER_APPS_REQUIRED_MODULES
  niftkOpenCV
)

# This sets up all the include and library folders using the above macro.
FOREACH(MOD ${HELPER_APPS_REQUIRED_MODULES})
  NIFTK_ADD_MODULE(${MOD})
ENDFOREACH()

# Then foreach app, we just build against libraries (modules).
FOREACH(APP ${HELPER_APPS})
  ADD_EXECUTABLE(${APP} ${APP}.cpp )
  TARGET_LINK_LIBRARIES(${APP} ${HELPER_APP_LIBRARIES} )
  MITK_INSTALL(TARGETS ${APP})
ENDFOREACH()
