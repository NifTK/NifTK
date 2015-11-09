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

set(version "4500dfe8c4")
set(location "${NIFTK_EP_TARBALL_LOCATION}/NifTK-MITK-${version}.tar.gz")

niftkMacroDefineExternalProjectVariables(MITK ${version} ${location})
set(proj_DEPENDENCIES ITK VTK GDCM DCMTK Eigen)
if(QT_FOUND)
  list(APPEND proj_DEPENDENCIES CTK)
endif(QT_FOUND)
if(BUILD_IGI)
  list(APPEND proj_DEPENDENCIES OpenCV NiftyLink)
  if(BUILD_PCL)
    list(APPEND proj_DEPENDENCIES FLANN PCL)
  endif()
endif(BUILD_IGI)

# explicitly try to tame windows headers.
if(WIN32)
  set(MITK_ADDITIONAL_C_FLAGS ${MITK_ADDITIONAL_C_FLAGS} "-DNOMINMAX")
  set(MITK_ADDITIONAL_CXX_FLAGS ${MITK_ADDITIONAL_CXX_FLAGS} "-DNOMINMAX")
endif(WIN32)


if(NOT DEFINED MITK_DIR)

    # Configure the MITK Superbuild, to decide which modules and plugins we want.
    #
    # Listing a module or plugin here does not mean that it will actually be built. That you can
    # control through CMake flags when configuring MITK. However, if a required module or plugin
    # is not listed here, it will not be built and it might prevent other modules or plugins from
    # being built that you would need. (If the required module depends on a module that is not
    # listed, eventually transitively.) The dependency lists are not complete, only the first
    # dependency is marked in the comments.

    set(_enabled_modules "")
    set(_enabled_plugins "")

    if(NIFTK_Apps/NiftyView)

      list(APPEND _enabled_modules
      # Modules with linking dependency from our code:
        Core                    # needed by niftkCore
        SceneSerializationBase  # needed by niftkCoreIO
        LegacyGL                # needed by PlanarFigure and AlgorithmsExt
        PlanarFigure            # needed by QtWidgets
        Overlays                # needed by QtWidgets
        QtWidgets               # needed by niftkCoreGui
        DataTypesExt            # needed by AlgorithmsExt
        AlgorithmsExt           # needed by ImageExtraction
        ImageExtraction         # needed by ImageStatistics
        ImageStatistics         # needed by QtWidgetsExt
        QtWidgetsExt            # needed by niftkCoreGui
        SceneSerialization      # needed by org.mitk.gui.qt.ext
        AppUtil                 # needed by org.mitk.gui.qt.ext
        LegacyIO                # needed by uk.ac.ucl.cmic.xnat
        LegacyAdaptors          # needed by Segmentation
        SurfaceInterpolation    # needed by Segmentation
        GraphAlgorithms         # needed by Segmentation
        ContourModel            # needed by Segmentation
        Multilabel              # needed by Segmentation
        Segmentation            # needed by uk.ac.ucl.cmic.surfaceextractor
        MapperExt               # needed by org.mitk.gui.qt.basicimageprocessing
        ImageDenoising          # needed by org.mitk.gui.qt.basicimageprocessing
        SegmentationUI          # needed by org.mitk.gui.qt.segmentation
        DicomUI                 # needed by org.mitk.gui.qt.dicom
        Python                  # needed by org.mitk.gui.qt.python
      # Auto-load modules. No linking dependency, but they provide IO classes and mappers that we need.
        IOExt
      #  IpPicSupportIO
      #  VtkShaders
      )

      list(APPEND _enabled_plugins
      # Plugins with compile-time or run-time dependency from our code:
        org.blueberry.core.runtime      # needed by org.blueberry.core.commands
        org.blueberry.core.commands     # needed by org.blueberry.ui.qt
        org.blueberry.core.expressions  # needed by org.blueberry.ui.qt
        org.blueberry.ui.qt             # needed by NiftyView
        org.mitk.core.ext               # needed by org.mitk.gui.qt.ext
        org.mitk.core.services          # needed by org.mitk.gui.common
        org.mitk.gui.common             # needed by org.mitk.gui.qt.application
        org.mitk.gui.qt.application     # needed by org.mitk.gui.qt.datamanager and org.mitk.gui.qt.ext
        org.mitk.gui.qt.common          # needed by org.mitk.gui.qt.datamanager
        org.mitk.gui.qt.ext             # needed by uk.ac.ucl.cmic.gui.qt.commonapps
        org.mitk.gui.qt.datamanager     # needed by NiftyView
      # Plugins that we do not physically depend on but that we want to use:
        org.blueberry.ui.qt.help
        org.blueberry.ui.qt.log
        org.mitk.planarfigure
        org.mitk.gui.qt.common.legacy           # needed by org.mitk.gui.qt.basicimageprocessing
        org.mitk.gui.qt.imagenavigator
        org.mitk.gui.qt.basicimageprocessing
        org.mitk.gui.qt.volumevisualization
        org.mitk.gui.qt.pointsetinteraction
        org.mitk.gui.qt.stdmultiwidgeteditor
        org.mitk.gui.qt.segmentation
        org.mitk.gui.qt.cmdlinemodules
        org.mitk.gui.qt.dicom
        org.mitk.gui.qt.measurementtoolbox
        org.mitk.gui.qt.moviemaker
        org.mitk.gui.qt.properties
        org.mitk.gui.qt.python
      )

    endif()

    if(NIFTK_Apps/NiftyIGI)

      list(APPEND _enabled_modules
      # Modules with linking dependency from our code:
        Core                    # needed by niftkCore
        SceneSerializationBase  # needed by niftkCoreIO
        LegacyGL                # needed by PlanarFigure and AlgorithmsExt
        PlanarFigure            # needed by QtWidgets
        Overlays                # needed by QtWidgets
        QtWidgets               # needed by niftkCoreGui
        DataTypesExt            # needed by AlgorithmsExt
        AlgorithmsExt           # needed by ImageExtraction
        ImageExtraction         # needed by ImageStatistics
        ImageStatistics         # needed by QtWidgetsExt
        QtWidgetsExt            # needed by niftkCoreGui
        SceneSerialization      # needed by org.mitk.gui.qt.ext
        AppUtil                 # needed by org.mitk.gui.qt.ext
        LegacyIO                # needed by uk.ac.ucl.cmic.xnat
        LegacyAdaptors          # needed by Segmentation
        SurfaceInterpolation    # needed by Segmentation
        GraphAlgorithms         # needed by Segmentation
        ContourModel            # needed by Segmentation
        Multilabel              # needed by Segmentation
        Segmentation            # needed by uk.ac.ucl.cmic.surfaceextractor
        IGTBase                 # needed by IGT
        OpenIGTLink             # needed by IGT
        IGT                     # needed by CameraCalibration
        OpenCVVideoSupport      # needed by niftkOpenCV
        CameraCalibration       # needed by niftkOpenCV
        Persistence             # needed by IGTUI
        IGTUI                   # needed by niftkIGIGui
      # Auto-load modules. No linking dependency, but they provide IO classes and mappers that we need.
        IOExt
      #  IpPicSupportIO
      #  VtkShaders
      )

      list(APPEND _enabled_plugins
      # Plugins with compile-time or run-time dependency from our code:
        org.blueberry.core.runtime      # needed by org.blueberry.core.commands
        org.blueberry.core.commands     # needed by org.blueberry.ui.qt
        org.blueberry.core.expressions  # needed by org.blueberry.ui.qt
        org.blueberry.ui.qt             # needed by NiftyIGI
        org.mitk.core.ext               # needed by org.mitk.gui.qt.ext
        org.mitk.core.services          # needed by org.mitk.gui.common
        org.mitk.gui.common             # needed by org.mitk.gui.qt.application
        org.mitk.gui.qt.application     # needed by org.mitk.gui.qt.datamanager and org.mitk.gui.qt.ext
        org.mitk.gui.qt.common          # needed by org.mitk.gui.qt.datamanager
        org.mitk.gui.qt.ext             # needed by NiftyIGI
        org.mitk.gui.qt.datamanager     # needed by NiftyIGI
      # Plugins that we do not physically depend on but that we want to use:
        org.blueberry.ui.qt.help
        org.blueberry.ui.qt.log
        org.mitk.planarfigure
        org.mitk.gui.qt.imagenavigator
        org.mitk.gui.qt.properties
        org.mitk.gui.qt.aicpregistration
      )

    endif()

    if(MITK_USE_Python)

      list(APPEND _enabled_modules
        Python                  # needed by org.mitk.gui.qt.python
      )

      list(APPEND _enabled_plugins
        org.mitk.gui.qt.python
      )

    endif()

    list(REMOVE_DUPLICATES _enabled_modules)
    list(REMOVE_DUPLICATES _enabled_plugins)

    set(_whitelists_dir "${CMAKE_CURRENT_BINARY_DIR}")
    set(_whitelist_name "MITK-whitelist")
    file(WRITE "${_whitelists_dir}/${_whitelist_name}.cmake" "
      set(enabled_modules
        ${_enabled_modules}
      )
      set(enabled_plugins
        ${_enabled_plugins}
      )
    ")

    # Note:
    # The DCMTK_DIR variable should not really be set here. This is a workaround because
    # the variable gets overwritten in MITKConfig.cmake from the DCMTK install directory
    # to the directory that contains DCMTKConfig.cmake (.../share/dcmtk).

    set(MITK_INITIAL_CACHE_FILE "${CMAKE_CURRENT_BINARY_DIR}/mitk_initial_cache.txt")
    file(WRITE "${MITK_INITIAL_CACHE_FILE}" "
      set(MITK_BUILD_APP_CoreApp OFF CACHE BOOL \"Build the MITK CoreApp application. This should be OFF, as NifTK has it's own application NiftyView. \")
      set(MITK_BUILD_APP_mitkWorkbench OFF CACHE BOOL \"Build the MITK Workbench application. This should be OFF, as NifTK has it's own application NiftyView. \")
      set(MITK_BUILD_APP_mitkDiffusion OFF CACHE BOOL \"Build the MITK Diffusion application. This should be OFF, as NifTK has it's own application NiftyView. \")
      set(MITK_BUILD_APP_Workbench OFF CACHE BOOL \"Build the MITK Workbench application. This should be OFF, as NifTK has it's own application NiftyView. \")
      set(MITK_BUILD_APP_Diffusion OFF CACHE BOOL \"Build the MITK Diffusion application. This should be OFF, as NifTK has it's own application NiftyView. \")
      set(MITK_BUILD_org.mitk.gui.qt.application ON CACHE BOOL \"Build the MITK application plugin. This should be ON, as it contains support classes we need for NiftyView. \")
      set(MITK_BUILD_org.mitk.gui.qt.ext ON CACHE BOOL \"Build the MITK ext plugin. This should be ON, as it contains support classes we need for NiftyView. \")
      set(MITK_BUILD_org.mitk.gui.qt.extapplication OFF CACHE BOOL \"Build the MITK ExtApp plugin. This should be OFF, as NifTK has it's own application NiftyView. \")
      set(MITK_BUILD_org.mitk.gui.qt.coreapplication OFF CACHE BOOL \"Build the MITK CoreApp plugin. This should be OFF, as NifTK has it's own application NiftyView. \")
      set(MITK_BUILD_org.mitk.gui.qt.imagecropper OFF CACHE BOOL \"Build the MITK image cropper plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.measurement OFF CACHE BOOL \"Build the MITK measurement plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.basicimageprocessing ON CACHE BOOL \"Build the MITK basic image processing tools\") 
      set(MITK_BUILD_org.mitk.gui.qt.volumevisualization ON CACHE BOOL \"Build the MITK volume visualization plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.pointsetinteraction ON CACHE BOOL \"Build the MITK point set interaction plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.stdmultiwidgeteditor ON CACHE BOOL \"Build the MITK ortho-viewer plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.segmentation ON CACHE BOOL \"Build the MITK segmentation plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.cmdlinemodules ON CACHE BOOL \"Build the MITK Command Line Modules plugin. \")
      set(MITK_BUILD_org.mitk.gui.qt.dicom ON CACHE BOOL \"Build the MITK DICOM plugin. \")
      set(MITK_BUILD_org.mitk.gui.qt.measurementtoolbox ON CACHE BOOL \"Build the MITK measurement toolbox, but we turn the statistics plugin off in the C++ code. \")
      set(MITK_BUILD_org.mitk.gui.qt.moviemaker ON CACHE BOOL \"Build the MITK Movie Maker plugin. \")
      set(MITK_BUILD_org.mitk.gui.qt.aicpregistration ON CACHE BOOL \"Build the MITK Anisotropic ICP plugin. \")
      set(MITK_BUILD_org.mitk.gui.qt.python ${MITK_USE_Python} CACHE BOOL \"Build the MITK python plugin. \")
      set(MITK_BUILD_org.blueberry.ui.qt ON CACHE BOOL \"Build the org.blueberry.ui.qt plugin\")
      set(MITK_BUILD_org.blueberry.ui.qt.log ON CACHE BOOL \"Build the Blueberry logging plugin\")
      set(MITK_BUILD_org.blueberry.ui.qt.help ON CACHE BOOL \"Build the Blueberry Qt help plugin\")
      set(DCMTK_DIR ${DCMTK_DIR} CACHE PATH \"DCMTK install directory\")
      set(Python_DIR ${Python_DIR} CACHE PATH \"Python install directory \")
    ")

    set(mitk_optional_cache_args )
    if(MITK_USE_Python)
      list(APPEND mitk_optional_cache_args
           -DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}
           -DPYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR}
           -DPYTHON_LIBRARY:FILEPATH=${PYTHON_LIBRARY}
           -DPYTHON_INCLUDE_DIR2:PATH=${PYTHON_INCLUDE_DIR2}
           -DPython_DIR:PATH=${Python_DIR}
           -DMITK_USE_SYSTEM_PYTHON:BOOL=${MITK_USE_SYSTEM_PYTHON}
           -DMITK_USE_Python:BOOL=${MITK_USE_Python}
          )
      list(APPEND proj_DEPENDENCIES Python)
      foreach(dep ZLIB PCRE SWIG SimpleITK Numpy)
        if(${MITK_USE_${dep}})
          list(APPEND proj_DEPENDENCIES ${dep})
          list(APPEND mitk_optional_cache_args -DMITK_USE_${dep}:BOOL=${MITK_USE_${dep}} -D${dep}_DIR:PATH=${${dep}_DIR} )
        endif()
      endforeach()
    endif()

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      UPDATE_COMMAND  ${GIT_EXECUTABLE} checkout ${proj_VERSION}
      INSTALL_COMMAND ""
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DMITK_BUILD_TUTORIAL:BOOL=OFF
        -DMITK_BUILD_ALL_PLUGINS:BOOL=OFF
        -DMITK_USE_QT:BOOL=${QT_FOUND}
        -DMITK_USE_CTK:BOOL=${QT_FOUND}
        -DMITK_USE_BLUEBERRY:BOOL=${QT_FOUND}
        -DMITK_USE_GDCMIO:BOOL=ON
        -DMITK_USE_DCMTK:BOOL=ON
        -DMITK_USE_Boost:BOOL=OFF
        -DMITK_USE_OpenCV:BOOL=${BUILD_IGI}
        -DMITK_USE_OpenIGTLink:BOOL=${BUILD_IGI}
        -DMITK_USE_OpenCL:BOOL=${BUILD_VL}
        -DMITK_ADDITIONAL_C_FLAGS:STRING=${MITK_ADDITIONAL_C_FLAGS}
        -DMITK_ADDITIONAL_CXX_FLAGS:STRING=${MITK_ADDITIONAL_CXX_FLAGS}
        -DGDCM_DIR:PATH=${GDCM_DIR}                            # FindGDCM expects GDCM_DIR
        -DVTK_DIR:PATH=${VTK_DIR}                              # FindVTK expects VTK_DIR
        -DITK_DIR:PATH=${ITK_DIR}                              # FindITK expects ITK_DIR
        -DCTK_DIR:PATH=${CTK_DIR}                              # FindCTK expects CTK_DIR
        -DDCMTK_DIR:PATH=${DCMTK_DIR}                          # FindDCMTK expects DCMTK_DIR
        -DOpenCV_DIR:PATH=${OpenCV_DIR}
        -DOpenIGTLink_DIR:PATH=${OpenIGTLink_DIR}
        -DEigen_DIR:PATH=${Eigen_DIR}
        ${mitk_optional_cache_args}
        -DMITK_INITIAL_CACHE_FILE:FILEPATH=${MITK_INITIAL_CACHE_FILE}
        -DMITK_WHITELIST:STRING=${_whitelist_name}\ \(external\)
        -DMITK_WHITELISTS_EXTERNAL_PATH:STRING=${_whitelists_dir}
      CMAKE_CACHE_ARGS
        ${EP_COMMON_CACHE_ARGS}
      CMAKE_CACHE_DEFAULT_ARGS
        ${EP_COMMON_CACHE_DEFAULT_ARGS}
      DEPENDS ${proj_DEPENDENCIES}
    )
    set(MITK_DIR ${proj_BUILD}/${proj}-build)

#    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})
    mitkFunctionInstallExternalCMakeProject(${proj})

    message("SuperBuild loading MITK from ${MITK_DIR}")

else()

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif()
