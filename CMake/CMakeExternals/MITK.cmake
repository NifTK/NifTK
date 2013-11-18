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


#-----------------------------------------------------------------------------
# MITK
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED MITK_DIR AND NOT EXISTS ${MITK_DIR})
  message(FATAL_ERROR "MITK_DIR variable is defined but corresponds to non-existing directory \"${MITK_DIR}\".")
endif()

set(proj MITK)
set(proj_DEPENDENCIES Boost ITK VTK GDCM DCMTK)
if(QT_FOUND)
  list(APPEND proj_DEPENDENCIES CTK)
endif(QT_FOUND)
if(BUILD_IGI)
  list(APPEND proj_DEPENDENCIES aruco OpenCV Eigen apriltags)
  if(BUILD_PCL)
    list(APPEND proj_DEPENDENCIES FLANN PCL)
  endif()
endif(BUILD_IGI)

set(MITK_DEPENDS ${proj})

# explicitly try to tame windows headers.
if(WIN32)
  set(MITK_ADDITIONAL_C_FLAGS ${MITK_ADDITIONAL_C_FLAGS} "-DNOMINMAX")
  set(MITK_ADDITIONAL_CXX_FLAGS ${MITK_ADDITIONAL_CXX_FLAGS} "-DNOMINMAX")
endif(WIN32)

if(NOT DEFINED MITK_DIR)

    ######################################################################
    # Configure the MITK Superbuild, to decide which plugins we want.
    ######################################################################

    set(MITK_INITIAL_CACHE_FILE "${CMAKE_CURRENT_BINARY_DIR}/mitk_initial_cache.txt")
    file(WRITE "${MITK_INITIAL_CACHE_FILE}" "
      set(MITK_BUILD_APP_CoreApp OFF CACHE BOOL \"Build the MITK CoreApp application. This should be OFF, as NifTK has it's own application NiftyView. \")
      set(MITK_BUILD_APP_mitkWorkbench OFF CACHE BOOL \"Build the MITK Workbench application. This should be OFF, as NifTK has it's own application NiftyView. \")
      set(MITK_BUILD_APP_mitkDiffusion OFF CACHE BOOL \"Build the MITK Diffusion application. This should be OFF, as NifTK has it's own application NiftyView. \")      
      set(MITK_BUILD_org.mitk.gui.qt.application ON CACHE BOOL \"Build the MITK application plugin. This should be ON, as it contains support classes we need for NiftyView. \")
      set(MITK_BUILD_org.mitk.gui.qt.ext ON CACHE BOOL \"Build the MITK ext plugin. This should be ON, as it contains support classes we need for NiftyView. \")
      set(MITK_BUILD_org.mitk.gui.qt.extapplication OFF CACHE BOOL \"Build the MITK ExtApp plugin. This should be OFF, as NifTK has it's own application NiftyView. \")      
      set(MITK_BUILD_org.mitk.gui.qt.coreapplication OFF CACHE BOOL \"Build the MITK CoreApp plugin. This should be OFF, as NifTK has it's own application NiftyView. \")      
      set(MITK_BUILD_org.mitk.gui.qt.imagecropper OFF CACHE BOOL \"Build the MITK image cropper plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.measurement OFF CACHE BOOL \"Build the MITK measurement plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.volumevisualization ON CACHE BOOL \"Build the MITK volume visualization plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.pointsetinteraction ON CACHE BOOL \"Build the MITK point set interaction plugin\")            
      set(MITK_BUILD_org.mitk.gui.qt.stdmultiwidgeteditor ON CACHE BOOL \"Build the MITK ortho-viewer plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.segmentation OFF CACHE BOOL \"Build the MITK segmentation plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.cmdlinemodules ON CACHE BOOL \"Build the MITK Command Line Modules plugin. \")
      set(MITK_BUILD_org.mitk.gui.qt.dicom ON CACHE BOOL \"Build the MITK DICOM plugin. \")
      set(MITK_BUILD_org.mitk.gui.qt.measurementtoolbox ON CACHE BOOL \"Build the MITK measurement toolbox, but we turn the statistics plugin off in the C++ code. \")
      set(BLUEBERRY_BUILD_org.blueberry.ui.qt.log ON CACHE BOOL \"Build the Blueberry logging plugin\")
      set(BLUEBERRY_BUILD_org.blueberry.ui.qt.help ON CACHE BOOL \"Build the Blueberry Qt help plugin\")
      set(BLUEBERRY_BUILD_org.blueberry.compat ON CACHE BOOL \"Build the Blueberry compat plugin (Matt, what is this for?)\")
      set(BOOST_INCLUDEDIR ${BOOST_INCLUDEDIR} CACHE PATH \"Path to Boost include directory\")
      set(BOOST_LIBRARYDIR ${BOOST_LIBRARYDIR} CACHE PATH \"Path to Boost library directory\")
      set(DCMTK_DIR ${DCMTK_DIR} CACHE PATH \"Path to DCMTK installation directory\")
    ")

    niftkMacroGetChecksum(NIFTK_CHECKSUM_MITK ${NIFTK_LOCATION_MITK})

    set(mitk_additional_library_search_paths)
    if(BUILD_IGI)
      list(APPEND mitk_additional_library_search_paths ${aruco_DIR}/lib ${apriltags_LIBRARY_DIRS} ${FLANN_DIR}/lib ${PCL_DIR}/lib)
    endif()
    
    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj}-src
      BINARY_DIR ${proj}-build
      PREFIX ${proj}-cmake
      INSTALL_DIR ${proj}-install
      URL ${NIFTK_LOCATION_MITK}
      URL_MD5 ${NIFTK_CHECKSUM_MITK}
      UPDATE_COMMAND  ${GIT_EXECUTABLE} checkout ${NIFTK_VERSION_MITK}
      INSTALL_COMMAND ""
      CMAKE_GENERATOR ${GEN}
      CMAKE_CACHE_ARGS
        ${EP_COMMON_ARGS}
        -DDESIRED_QT_VERSION:STRING=4
        -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
        -DMITK_BUILD_TUTORIAL:BOOL=OFF
        -DMITK_BUILD_ALL_PLUGINS:BOOL=OFF
        -DMITK_USE_QT:BOOL=${QT_FOUND}
        -DMITK_USE_CTK:BOOL=${QT_FOUND}
        -DMITK_USE_BLUEBERRY:BOOL=${QT_FOUND}
        -DMITK_USE_GDCMIO:BOOL=ON
        -DMITK_USE_DCMTK:BOOL=ON
        -DMITK_USE_Boost:BOOL=ON
        -DMITK_USE_Boost_LIBRARIES:STRING="filesystem system date_time"
        -DMITK_USE_SYSTEM_Boost:BOOL=OFF
        -DMITK_USE_OpenCV:BOOL=${BUILD_IGI}
        -DMITK_ADDITIONAL_C_FLAGS:STRING=${MITK_ADDITIONAL_C_FLAGS}
        -DMITK_ADDITIONAL_CXX_FLAGS:STRING=${MITK_ADDITIONAL_CXX_FLAGS}
        -DMITK_ADDITIONAL_LIBRARY_SEARCH_PATHS:STRING=${mitk_additional_library_search_paths}
        -DEXTERNAL_BOOST_ROOT:PATH=${BOOST_ROOT}               # FindBoost expects BOOST_ROOT
        -DBOOST_INCLUDEDIR:PATH=${BOOST_INCLUDEDIR}            # Derived from BOOST_ROOT, set in BOOST.cmake
        -DBOOST_LIBRARYDIR:PATH=${BOOST_LIBRARYDIR}            # Derived from BOOST_ROOT, set in BOOST.cmake
        -DGDCM_DIR:PATH=${GDCM_DIR}                            # FindGDCM expects GDCM_DIR
        -DVTK_DIR:PATH=${VTK_DIR}                              # FindVTK expects VTK_DIR
        -DITK_DIR:PATH=${ITK_DIR}                              # FindITK expects ITK_DIR
        -DCTK_DIR:PATH=${CTK_DIR}                              # FindCTK expects CTK_DIR
        -DDCMTK_DIR:PATH=${DCMTK_DIR}                          # FindDCMTK expects DCMTK_DIR
        -DOpenCV_DIR:PATH=${OpenCV_DIR}
        -DMITK_INITIAL_CACHE_FILE:FILEPATH=${MITK_INITIAL_CACHE_FILE}
      DEPENDS ${proj_DEPENDENCIES}
      )
    set(MITK_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build/${proj}-build)
    message("SuperBuild loading MITK from ${MITK_DIR}")

else()

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif()
