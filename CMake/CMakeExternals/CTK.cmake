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
  SET(proj_DEPENDENCIES VTK ITK)
  SET(CTK_DEPENDS ${proj})
  
  IF(NOT DEFINED CTK_DIR)
  
      SET(revision_tag 6f26c34)
      IF(${proj}_REVISION_TAG)
        SET(revision_tag ${${proj}_REVISION_TAG})
      ENDIF()
      
      ExternalProject_Add(${proj}
      GIT_REPOSITORY http://github.com/commontk/CTK.git
      GIT_TAG ${revision_tag}
      BINARY_DIR ${proj}-build
      UPDATE_COMMAND ""
      INSTALL_COMMAND ""
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DDESIRED_QT_VERSION:STRING=4
        -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
        -DCTK_LIB_PluginFramework:BOOL=ON
        -DCTK_LIB_DICOM/Widgets:BOOL=ON
        -DCTK_PLUGIN_org.commontk.eventadmin:BOOL=ON
        -DCTK_USE_GIT_PROTOCOL:BOOL=OFF
        -DADDITIONAL_C_FLAGS:STRING=${NIFTK_ADDITIONAL_C_FLAGS}
        -DADDITIONAL_CXX_FLAGS:STRING=${NIFTK_ADDITIONAL_CXX_FLAGS}
        -DVTK_DIR:PATH=${VTK_DIR}                              # FindVTK expects VTK_DIR
        -DITK_DIR:PATH=${ITK_DIR}                              # FindITK expects ITK_DIR
      DEPENDS ${proj_DEPENDENCIES}
    )
  SET(CTK_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)
  MESSAGE("SuperBuild loading CTK from ${CTK_DIR}")
  
  ELSE()
  
    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
  
  ENDIF()

ENDIF(QT_FOUND)
