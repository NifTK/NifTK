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

SET(HELPER_APPS_REQUIRED_MODULES
  # Add modules that DO NOT depend on BUILD_IGI here.
)

IF(BUILD_IGI)
  SET(HELPER_APPS_REQUIRED_MODULES
    ${HELPER_APPS_REQUIRED_MODULES}
    niftkOpenCV
  )  
ENDIF()

# This sets up all the include and library folders using the above macro.
FOREACH(MOD ${HELPER_APPS_REQUIRED_MODULES})
  NIFTK_ADD_MODULE(${MOD})
ENDFOREACH()

SET(HELPER_APPS
  # Add apps that DO NOT depend on BUILD_IGI here.
)

IF(BUILD_IGI)
  SET(HELPER_APPS
    ${HELPER_APPS}
    # Add apps that DO depend on BUILD_IGI here.
    niftkOpenCVTest
  )
ENDIF()

# Then foreach app, we just build against libraries (modules).
FOREACH(APP ${HELPER_APPS})
  ADD_EXECUTABLE(${APP} ${APP}.cxx )
  TARGET_LINK_LIBRARIES(${APP} ${HELPER_APP_LIBRARIES} )
  MITK_INSTALL(TARGETS ${APP})
ENDFOREACH()

# Repeat the same process for slices apps.
IF(SlicerExecutionModel_FOUND)
  SET(SLICER_HELPER_APPS
    # Add Slicer apps that DO NOT depend on BUILD_IGI here.
  )

  IF(BUILD_IGI)
    SET(SLICER_HELPER_APPS
      ${SLICER_HELPER_APPS}
      niftkCameraCalibration
      niftkCorrectVideoDistortion
      niftkTrackLapUSProbe
      niftkProject3DPointsToStereoPair
    )
  ENDIF()  

  FOREACH(APP ${SLICER_HELPER_APPS})    
    NIFTK_CONFIGURE_AND_BUILD_CLI(
      NAME ${APP} 
      TARGET_LIBRARIES ${HELPER_APP_LIBRARIES}
    )
  ENDFOREACH()
ENDIF()
