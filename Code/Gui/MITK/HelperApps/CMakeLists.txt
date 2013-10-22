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

macro(NIFTK_ADD_MODULE module_name)
  include(${${PROJECT_NAME}_MODULES_CONF_DIR}/${module_name}Config.cmake)
  include_directories(${${module_name}_INCLUDE_DIRS})
  link_directories(${${module_name}_LIBRARY_DIRS})
  set(HELPER_APP_LIBRARIES
    ${HELPER_APP_LIBRARIES}
    ${${module_name}_PROVIDES}
    ${${module_name}_DEPENDS}    
  )
endmacro()

link_directories(${MITK_LIBRARY_DIRS})
include_directories(${MITK_INCLUDE_DIRS})
include_directories(${${PROJECT_NAME}_MODULES_CONF_DIR})

set(HELPER_APP_LIBRARIES
  ${ALL_LIBRARIES}
)

set(HELPER_APPS_REQUIRED_MODULES
  # Add modules that DO NOT depend on BUILD_IGI here.
)

if(BUILD_IGI)
  set(HELPER_APPS_REQUIRED_MODULES
    ${HELPER_APPS_REQUIRED_MODULES}
    niftkOpenCV
  )  
endif()

# This sets up all the include and library folders using the above macro.
foreach(MOD ${HELPER_APPS_REQUIRED_MODULES})
  NIFTK_ADD_MODULE(${MOD})
endforeach()

set(HELPER_APPS
  # Add apps that DO NOT depend on BUILD_IGI here.
)

if(BUILD_IGI)
  set(HELPER_APPS
    ${HELPER_APPS}
    # Add apps that DO depend on BUILD_IGI here.
  )
endif()

# Then foreach app, we just build against libraries (modules).
foreach(APP ${HELPER_APPS})
  NIFTK_CREATE_COMMAND_LINE_APPLICATION(
    NAME ${APP}
    BUILD_CLI
    TARGET_LIBRARIES ${HELPER_APP_LIBRARIES}
  )
endforeach()

# Repeat the same process for slicer apps.
if(SlicerExecutionModel_FOUND)
  set(SLICER_HELPER_APPS
    # Add Slicer apps that DO NOT depend on BUILD_IGI here.
  )

  if(BUILD_IGI)
    set(SLICER_HELPER_APPS
      ${SLICER_HELPER_APPS}
      niftkCameraCalibration
      niftkCorrectVideoDistortion
      niftkProject3DPointsToStereoPair
      niftkHandeyeCalibration
      niftkHandeyeCalibrationFromDirectory
      niftkGetMatchedTrackingMatrix
      niftkLaparoscopeTrackingAccuracy
      niftkProjectTrackedPointsOnStereoVideo
      niftkFindAndTriangulateCrossHair
      niftkUltrasoundPinCalibration
      niftkTriangulate2DPointPairsTo3D
      niftkPivotCalibration
      niftkMakeGridOf2DImages
    )
  endif()  

  foreach(APP ${SLICER_HELPER_APPS})
    NIFTK_CREATE_COMMAND_LINE_APPLICATION(
      NAME ${APP}
      BUILD_SLICER
      INSTALL_SCRIPT
      TARGET_LIBRARIES ${HELPER_APP_LIBRARIES}
    )
  endforeach()
endif()
