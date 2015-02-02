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

  niftkMacroDefineExternalProjectVariables(CTK ${NIFTK_VERSION_CTK})
  set(proj_DEPENDENCIES VTK ITK DCMTK)

  if(NOT DEFINED CTK_DIR)

    niftkMacroGetChecksum(NIFTK_CHECKSUM_CTK ${NIFTK_LOCATION_CTK})

    set (ctk_qt_args -DCTK_QT_VERSION:STRING=${DESIRED_QT_VERSION})

    if (DESIRED_QT_VERSION MATCHES "5")
      list(APPEND ctk_qt_args -DQT5_INSTALL_PREFIX:FILEPATH=${QT5_INSTALL_PREFIX})
    else()
      list(APPEND ctk_qt_args -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE})
    endif()

    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj_SOURCE}
      PREFIX ${proj_CONFIG}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${NIFTK_LOCATION_CTK}
      URL_MD5 ${NIFTK_CHECKSUM_CTK}
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${proj_VERSION}
      INSTALL_COMMAND ""
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
         ${ctk_qt_args}
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
        -DDCMTK_URL:STRING=http://cmic.cs.ucl.ac.uk/platform/dependencies/CTK_DCMTK_085525e6.tar.gz
        -DqRestAPI_URL:STRING=${NIFTK_LOCATION_qRestAPI}
      DEPENDS ${proj_DEPENDENCIES}
    )
    set(CTK_DIR ${proj_BUILD})
    set(CTK_SOURCE_DIR ${proj_SOURCE})

    message("SuperBuild loading CTK from ${CTK_DIR}")

  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()

endif(QT_FOUND)
