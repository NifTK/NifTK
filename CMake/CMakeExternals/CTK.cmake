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
# CTK. Note, we are building it ourselves, rather than rely on the MITK
# settings. This mean that the default MITK build may have different
# settings than what we are specifying here. So NIFTK and MITK may be out
# of sync. However, this gives us a bit more flexibility.
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED CTK_DIR AND NOT EXISTS ${CTK_DIR})
  message(FATAL_ERROR "CTK_DIR variable is defined but corresponds to non-existing directory \"${CTK_DIR}\"")
endif()

if(QT_FOUND)

  # Note: If the CTK version changes, then you either clear the plugin cache
  # or change the deploy path by changing the patch level.
  set(version "910cec7415")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/NifTK-CTK-${version}.tar.gz")

  set(qRestAPI_version "5f3a03b15d")
  set(qRestAPI_location "${NIFTK_EP_TARBALL_LOCATION}/commontk-qRestAPI-${qRestAPI_version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(CTK ${version} ${location})
  set(proj_DEPENDENCIES VTK ITK DCMTK)

  if(NOT DEFINED CTK_DIR)

    set(ctk_optional_cache_args )
    if(MITK_USE_Python)
      if(NOT MITK_USE_SYSTEM_PYTHON)
        list(APPEND proj_DEPENDENCIES Python)
      endif()
      list(APPEND ctk_optional_cache_args
           -DCTK_LIB_Scripting/Python/Widgets:BOOL=ON
           -DCTK_ENABLE_Python_Wrapping:BOOL=ON
           -DCTK_APP_ctkSimplePythonShell:BOOL=ON
           -DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}
           -DPYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR}
           -DPYTHON_INCLUDE_DIR2:PATH=${PYTHON_INCLUDE_DIR2}
           -DPYTHON_LIBRARY:FILEPATH=${PYTHON_LIBRARY}
      )
    else()
      list(APPEND ctk_optional_cache_args
           -DCTK_LIB_Scripting/Python/Widgets:BOOL=OFF
           -DCTK_ENABLE_Python_Wrapping:BOOL=OFF
           -DCTK_APP_ctkSimplePythonShell:BOOL=OFF
      )
    endif()

    if(MITK_USE_DCMTK)
      list(APPEND ctk_optional_cache_args
           -DDCMTK_DIR:PATH=${DCMTK_DIR}
          )
      if(NOT MITK_USE_Python)
        list(APPEND ctk_optional_cache_args
            -DDCMTK_CMAKE_DEBUG_POSTFIX:STRING=d
            )
      endif()
      list(APPEND proj_DEPENDENCIES DCMTK)
    endif()

    if(CTEST_USE_LAUNCHERS)
      list(APPEND ctk_optional_cache_args
        "-DCMAKE_PROJECT_${proj}_INCLUDE:FILEPATH=${CMAKE_ROOT}/Modules/CTestUseLaunchers.cmake"
      )
    endif()

    set (ctk_qt_args -DCTK_QT_VERSION:STRING=${DESIRED_QT_VERSION})

    if (DESIRED_QT_VERSION MATCHES "5")
      list(APPEND ctk_qt_args -DQT5_INSTALL_PREFIX:FILEPATH=${QT5_INSTALL_PREFIX})
    else()
      list(APPEND ctk_qt_args -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE})
    endif()

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${proj_VERSION}
      INSTALL_COMMAND ""
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        ${ctk_optional_cache_args}
        ${ctk_qt_args}
        # The CTK PluginFramework cannot cope with
        # a non-empty CMAKE_DEBUG_POSTFIX for the plugin
        # libraries yet.
        -DCMAKE_DEBUG_POSTFIX:STRING=
        -DGit_EXECUTABLE:FILEPATH=${GIT_EXECUTABLE}
        -DGIT_EXECUTABLE:FILEPATH=${GIT_EXECUTABLE}
        -DCTK_LIB_CommandLineModules/Backend/LocalProcess:BOOL=ON
        -DCTK_LIB_CommandLineModules/Frontend/QtGui:BOOL=ON
        -DCTK_LIB_PluginFramework:BOOL=ON
        -DCTK_LIB_DICOM/Widgets:BOOL=ON
        -DCTK_LIB_DICOM/Core:BOOL=OFF
        -DCTK_LIB_WIDGETS:BOOL=ON
        -DCTK_PLUGIN_org.commontk.eventadmin:BOOL=ON
        -DCTK_PLUGIN_org.commontk.configadmin:BOOL=ON
        # CTK ignores the other standard flags variables:
        #   CMAKE_*_FLAGS_DEBUG, CMAKE_*_FLAGS_RELEASE, CMAKE_*_FLAGS_RELWITHDEBINFO, CMAKE_*_LINKER_FLAGS
        -DADDITIONAL_C_FLAGS:STRING=${CTK_ADDITIONAL_C_FLAGS}
        -DADDITIONAL_CXX_FLAGS:STRING=${CTK_ADDITIONAL_CXX_FLAGS}
        -DCTK_LIB_XNAT/Core:BOOL=ON
        -DCTK_LIB_XNAT/Widgets:BOOL=ON
        -DDCMTK_DIR:PATH=${DCMTK_DIR}
        -DVTK_DIR:PATH=${VTK_DIR}
        -DITK_DIR:PATH=${ITK_DIR}
        -DDCMTK_URL:STRING=${NIFTK_EP_TARBALL_LOCATION}/CTK_DCMTK_085525e6.tar.gz
        -DqRestAPI_URL:STRING=${qRestAPI_location}
      CMAKE_CACHE_ARGS
        ${EP_COMMON_CACHE_ARGS}
      CMAKE_CACHE_DEFAULT_ARGS
        ${EP_COMMON_CACHE_DEFAULT_ARGS}
      DEPENDS ${proj_DEPENDENCIES}
    )
    set(CTK_DIR ${proj_BUILD})
    set(CTK_SOURCE_DIR ${proj_SOURCE})

    #set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})
    #mitkFunctionInstallExternalCMakeProject(${proj})

    message("SuperBuild loading CTK from ${CTK_DIR}")

  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()

endif(QT_FOUND)
