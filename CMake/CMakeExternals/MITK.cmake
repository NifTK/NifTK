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

set(version "f927d792f3")
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

    # Configure the MITK Superbuild to decide which modules and plugins we want.
    #
    # We control this via a whitelist. Any modules or plugins that are not listed
    # will not be built and will not even be configured, either. This reduces the
    # configuration time and build time, too. Note that since we do not build any
    # unnecessary plugins, there is no need to control which of them to enable
    # through the MITK initial cache options any more. We can simply switch on all
    # the available plugins.
    #
    # Note also that the whitelist must be transitively closed, i.e. if you need a
    # module or plugin then all of its dependencies must be on the whitelist, too.
    # The list of dependencies below are not complete, only the first dependency is
    # marked in the comments.

    set(_enabled_modules "")
    set(_enabled_plugins "")

    # Common requirements for GUI applications:
    if(NIFTK_Apps/NiftyView OR NIFTK_Apps/NiftyIGI)

      list(APPEND _enabled_modules
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
        IOExt                   # autoloaded with MitkCore, registers IO microservices
      )

      list(APPEND _enabled_plugins
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
        org.blueberry.ui.qt.help
        org.blueberry.ui.qt.log
        org.mitk.planarfigure
        org.mitk.gui.qt.imagenavigator
        org.mitk.gui.qt.properties
      )
    endif()

    # Additionally required for NiftyView:
    if(NIFTK_Apps/NiftyView)

      list(APPEND _enabled_modules
        MapperExt               # needed by org.mitk.gui.qt.basicimageprocessing
        ImageDenoising          # needed by org.mitk.gui.qt.basicimageprocessing
        SegmentationUI          # needed by org.mitk.gui.qt.segmentation
        DicomUI                 # needed by org.mitk.gui.qt.dicom
      )

      list(APPEND _enabled_plugins
        org.mitk.gui.qt.common.legacy           # needed by org.mitk.gui.qt.basicimageprocessing
        org.mitk.gui.qt.basicimageprocessing
        org.mitk.gui.qt.volumevisualization
        org.mitk.gui.qt.pointsetinteraction
        org.mitk.gui.qt.stdmultiwidgeteditor
        org.mitk.gui.qt.segmentation
        org.mitk.gui.qt.cmdlinemodules
        org.mitk.gui.qt.dicom
        org.mitk.gui.qt.measurementtoolbox
        org.mitk.gui.qt.moviemaker
      )

    endif()

    # Additionally required for NiftyIGI:
    if(NIFTK_Apps/NiftyIGI)

      list(APPEND _enabled_modules
        IGTBase                 # needed by IGT
        OpenIGTLink             # needed by IGT
        IGT                     # needed by CameraCalibration
        OpenCVVideoSupport      # needed by niftkOpenCV
        CameraCalibration       # needed by niftkOpenCV
        Persistence             # needed by IGTUI
        IGTUI                   # needed by niftkIGIGui
      )

      list(APPEND _enabled_plugins
        org.mitk.gui.qt.aicpregistration
      )

    endif()

    if(BUILD_VL)

      list(APPEND _enabled_modules
        IGTBase                 # needed by IGT
        OpenIGTLink             # needed by IGT
        IGT                     # needed by CameraCalibration
        OpenCVVideoSupport      # needed by niftkOpenCV
        CameraCalibration       # needed by niftkOpenCV
        OpenCL                  # needed by niftkVL
      )

    endif()

    if(BUILD_Python)

      list(APPEND _enabled_modules
        Python                  # needed by PythonService and QtPython
        PythonService           # autoloaded with Python
        QtPython                # needed by org.mitk.gui.qt.python
      )

      list(APPEND _enabled_plugins
        org.mitk.gui.qt.python
      )

    endif()

    list(REMOVE_DUPLICATES _enabled_modules)
    list(REMOVE_DUPLICATES _enabled_plugins)

    set(mitk_whitelists_dir "${CMAKE_CURRENT_BINARY_DIR}")
    set(mitk_whitelist_name "MITK-whitelist")
    file(WRITE "${mitk_whitelists_dir}/${mitk_whitelist_name}.cmake" "
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

    set(mitk_initial_cache_file "${CMAKE_CURRENT_BINARY_DIR}/MITK-initial_cache.txt")
    file(WRITE "${mitk_initial_cache_file}" "
      set(MITK_BUILD_APP_Workbench OFF CACHE BOOL \"Build the MITK Workbench application. This should be OFF, as NifTK has it's own application NiftyView. \")
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
        -DMITK_BUILD_ALL_PLUGINS:BOOL=ON
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
        -DEigen_INCLUDE_DIR:PATH=${Eigen_INCLUDE_DIR}
        ${mitk_optional_cache_args}
        -DMITK_INITIAL_CACHE_FILE:FILEPATH=${mitk_initial_cache_file}
        -DMITK_WHITELIST:STRING=${mitk_whitelist_name}\ \(external\)
        -DMITK_WHITELISTS_EXTERNAL_PATH:STRING=${mitk_whitelists_dir}
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
