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
  list(APPEND HELPER_APP_INCLUDE_DIRECTORIES ${${module_name}_INCLUDE_DIRS})
  list(APPEND HELPER_APP_LINK_DIRECTORIES ${${module_name}_LIBRARY_DIRS})
  list(APPEND HELPER_APP_LIBRARIES ${${module_name}_PROVIDES})
endmacro()

set(HELPER_APP_INCLUDE_DIRECTORIES)
set(HELPER_APP_LINK_DIRECTORIES)
set(HELPER_APP_LIBRARIES)

set(HELPER_APPS_REQUIRED_MODULES
  # Add modules that DO NOT depend on BUILD_IGI here.
)
if(BUILD_IGI)

  ##########################################################
  # mjc: Temporary? I can't find a way round including this.
  #      I suppose we could add it at the top level, but 
  #      the point of having it as a PackageDepends is so 
  #      that it is not at the top level.... hmmm.
  ##########################################################
  include(${PROJECT_SOURCE_DIR}/CMake/PackageDepends/MITK_NiftyLink_Config.cmake)

  list(APPEND HELPER_APPS_REQUIRED_MODULES 
    niftkOpenCV 
    niftkIGI
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
  )  
  if(BUILD_PCL)
    set(HELPER_APPS
      ${HELPER_APPS}
      niftkPointCloudWrite
    )    
  endif()
endif()

list(APPEND HELPER_APP_INCLUDE_DIRECTORIES  ${MITK_INCLUDE_DIRS} ${MITKEXT_INCLUDE_DIRS} ${${PROJECT_NAME}_MODULES_CONF_DIR})
list(APPEND HELPER_APP_LINK_DIRECTORIES     ${MITK_LIBRARY_DIRS} )

list(REMOVE_DUPLICATES HELPER_APP_INCLUDE_DIRECTORIES)
list(REMOVE_DUPLICATES HELPER_APP_LINK_DIRECTORIES)
list(REMOVE_DUPLICATES HELPER_APP_LIBRARIES)

include_directories(${HELPER_APP_INCLUDE_DIRECTORIES})
link_directories(${HELPER_APP_LINK_DIRECTORIES})

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
      niftkIterativeClosestPointRegister
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
