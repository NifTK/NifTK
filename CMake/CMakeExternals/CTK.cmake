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
#  Last Changed      : $LastChangedDate: 2011-12-17 14:35:07 +0000 (Sat, 17 Dec 2011) $ 
#  Revision          : $Revision: 8065 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

#-----------------------------------------------------------------------------
# CTK. Note, we are building it ourselves, rather than rely on the MITK
# settings. This mean that the default MITK build may have different 
# settings than what we are specifying here. So NIFTK and MITK may be out
# of sync. However, this gives us a bit more flexibility.
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED CTK_DIR AND NOT EXISTS ${CTK_DIR})
  MESSAGE(FATAL_ERROR "CTK_DIR variable is defined but corresponds to non-existing directory \"${CTK_DIR}\"")
ENDIF()

IF(QT_FOUND)

  SET(proj CTK)
  SET(proj_DEPENDENCIES VTK ITK DCMTK)
  SET(CTK_DEPENDS ${proj})
  
  IF(NOT DEFINED CTK_DIR)
  
      #####################################################################
      # Note: If the CTK version changes, then you either clear the plugin 
      # cache or change the deploy path by changing the patch level.
      #####################################################################
      SET(revision_tag c0d35ce9d)
      IF(${proj}_REVISION_TAG)
        SET(revision_tag ${${proj}_REVISION_TAG})
      ENDIF()
      
      ExternalProject_Add(${proj}
      SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj}-src
      BINARY_DIR ${proj}-build
      PREFIX ${proj}-cmake
      GIT_REPOSITORY ${GIT_PROTOCOL}:${NIFTK_LOCATION_CTK}
      GIT_TAG ${revision_tag}
      UPDATE_COMMAND ""
      INSTALL_COMMAND ""
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}       
        -DDESIRED_QT_VERSION:STRING=4
        -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
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
        -DCTK_USE_GIT_PROTOCOL:BOOL=${NIFTK_USE_GIT_PROTOCOL}
        -DADDITIONAL_C_FLAGS:STRING=${NIFTK_ADDITIONAL_C_FLAGS}
        -DADDITIONAL_CXX_FLAGS:STRING=${NIFTK_ADDITIONAL_CXX_FLAGS}
        -DDCMTK_DIR:PATH=${DCMTK_DIR}
        -DVTK_DIR:PATH=${VTK_DIR}
        -DITK_DIR:PATH=${ITK_DIR}
        -DDCMTK_URL:STRING=http://cmic.cs.ucl.ac.uk/platform/dependencies/CTK_DCMTK_085525e6.tar.gz 
      DEPENDS ${proj_DEPENDENCIES}
    )
  SET(CTK_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)
  SET(CTK_SOURCE_DIR  ${CMAKE_CURRENT_BINARY_DIR}/${proj}-src)

  MESSAGE("SuperBuild loading CTK from ${CTK_DIR}")
  
  ELSE()
  
    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
  
  ENDIF()

ENDIF(QT_FOUND)
