#-----------------------------------------------------------------------------
# MITK
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED MITK_DIR AND NOT EXISTS ${MITK_DIR})
  MESSAGE(FATAL_ERROR "MITK_DIR variable is defined but corresponds to non-existing directory \"${MITK_DIR}\".")
ENDIF()

SET(proj MITK)
SET(proj_DEPENDENCIES BOOST ITK VTK GDCM)  # Don't put CTK here, as it's optional, dependent on Qt.
IF(QT_FOUND)
  SET(proj_DEPENDENCIES BOOST ITK VTK GDCM CTK)
ENDIF(QT_FOUND)
SET(MITK_DEPENDS ${proj})

IF(NOT DEFINED MITK_DIR)

    ######################################################################
    # Configure the MITK Superbuild, to decide which plugins we want.
    ######################################################################

    set(MITK_INITIAL_CACHE_FILE "${CMAKE_CURRENT_BINARY_DIR}/mitk_initial_cache.txt")
    file(WRITE "${MITK_INITIAL_CACHE_FILE}" "
      set(MITK_BUILD_APP_CoreApp OFF CACHE BOOL \"Build the MITK CoreApp application. This should be OFF, as NifTK has it's own application NiftyView. \")
      set(MITK_BUILD_APP_ExtApp OFF CACHE BOOL \"Build the MITK ExtApp application. This should be OFF, as NifTK has it's own application NiftyView. \")
      set(MITK_BUILD_org.mitk.gui.qt.application OFF CACHE BOOL \"Build the MITK CoreApp plugin. This should be OFF, as NifTK has it's own application NiftyView. \")
      set(MITK_BUILD_org.mitk.gui.qt.extapplication OFF CACHE BOOL \"Build the MITK ExtApp plugin. This should be OFF, as NifTK has it's own application NiftyView. \")
      set(MITK_BUILD_org.mitk.gui.qt.ext ON CACHE BOOL \"Build the MITK ext plugin.\")
      set(MITK_BUILD_org.mitk.gui.qt.imagecropper OFF CACHE BOOL \"Build the MITK image cropper plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.measurement OFF CACHE BOOL \"Build the MITK measurement plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.pointsetinteraction OFF CACHE BOOL \"Build the MITK point set interaction plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.volumevisualization ON CACHE BOOL \"Build the MITK volume visualization plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.stdmultiwidgeteditor ON CACHE BOOL \"Build the MITK ortho-viewer plugin\")
      set(BLUEBERRY_BUILD_org.blueberry.ui.qt.log ON CACHE BOOL \"Build the Blueberry logging plugin\")
      set(BLUEBERRY_BUILD_org.blueberry.ui.qt.help ON CACHE BOOL \"Build the Blueberry Qt help plugin\")
      set(BLUEBERRY_BUILD_org.blueberry.compat ON CACHE BOOL \"Build the Blueberry compat plugin (Matt, what is this for?)\")
      set(BOOST_INCLUDEDIR ${BOOST_INCLUDEDIR} CACHE PATH \"Path to Boost include directory\")
      set(BOOST_LIBRARYDIR ${BOOST_LIBRARYDIR} CACHE PATH \"Path to Boost library directory\")
    ")

    #########################################################
    #
    # 1. Trac 1257, MITK revision f1953dbbb0, forked to branch niftk.
    #    The following changes were made, and merged in. 
    #
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/853  (opacity for black) 
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1255 (StateMachine.xml)
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1256 (improve file extension gz)
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1264 (Qt Assistant in installer)
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1300 (Enable drop in QmitkRenderWindow)
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1301 (Pass RenderingManager in QmitkStdMultiWidget constructor)
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1341 (Uniquely name QmitkStdMultiWidget planes)
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1342 (Stop RenderingManager eroneously adding to m_RenderWindowList)
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1343 (Fix AutoTopMost in mitkLevelWindowManager)
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1344 (Turn off interactors in mitkMouseModeSwitcher)
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1366 (Don't create PACS interactors in mitkMouseModeSwitcher)
    # 
    # 2. Trac 1379, merged MITK master version 8334d5b025 into niftk.
    #    Items in the list above, but not in the list below have already been 
    #    merged into MITK master branch, or are obselete. So, the effective codebase is 
    #    MITK 8334d5b025, plus the following list, resulting in niftk branch hashtag 5342dbafbc.
    #
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/853  (opacity for black. MITK working on proper fix).
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1256 (improve file extension gz. Not entirely merged ... needs re-checking).
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1344 (Turn off interactors in mitkMouseModeSwitcher)
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1366 (Don't create PACS interactors in mitkMouseModeSwitcher)
    #
    # 3. Trac 1379, merged MITK master cdaf95f4cd (20.04.2012 07:23) 
    #    to pick up MITK bug 11572, 11573, 11574, Jamies DICOM slice sorting 
    #    issue which is Trac 1365. Currently Im doing this on branch 
    #    NifTK-Trac-1379-MITK-upgrade, so 1365 will be done implicitly when 
    #    1379 complete. So, current code base is MITK cdaf95f4cd, plus 
    #    Trac 853, 1256, 1344 1366 as listed above, resulting in 
    #    2e5c698299 on MattClarkson github fork.
    # 
    #########################################################
    
    SET(revision_tag 2e5c698299)
    
    IF(${proj}_REVISION_TAG)
      SET(revision_tag ${${proj}_REVISION_TAG})
    ENDIF()
    
    ExternalProject_Add(${proj}
    GIT_REPOSITORY ${GIT_PROTOCOL}://github.com/MattClarkson/MITK.git
    GIT_TAG ${revision_tag}
    BINARY_DIR ${proj}-build
    UPDATE_COMMAND ""
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
      -DMITK_USE_OpenCV:BOOL=${BUILD_OPENCV}
      -DADDITIONAL_C_FLAGS:STRING=${NIFTK_ADDITIONAL_C_FLAGS}
      -DADDITIONAL_CXX_FLAGS:STRING=${NIFTK_ADDITIONAL_CXX_FLAGS}
      -DBOOST_ROOT:PATH=${BOOST_ROOT}                        # FindBoost expectes BOOST_ROOT  
      -DBOOST_INCLUDEDIR:PATH=${BOOST_INCLUDEDIR}            # Derived from BOOST_ROOT, set in BOOST.cmake
      -DBOOST_LIBRARYDIR:PATH=${BOOST_LIBRARYDIR}            # Derived from BOOST_ROOT, set in BOOST.cmake
      -DGDCM_DIR:PATH=${GDCM_DIR}                            # FindGDCM expects GDCM_DIR
      -DVTK_DIR:PATH=${VTK_DIR}                              # FindVTK expects VTK_DIR
      -DITK_DIR:PATH=${ITK_DIR}                              # FindITK expects ITK_DIR
      -DCTK_DIR:PATH=${CTK_DIR}                              # FindCTK expects CTK_DIR
      -DMITK_INITIAL_CACHE_FILE:FILEPATH=${MITK_INITIAL_CACHE_FILE}
    DEPENDS ${proj_DEPENDENCIES}
  )
SET(MITK_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build/${proj}-build)
MESSAGE("SuperBuild loading MITK from ${MITK_DIR}")

ELSE()

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

ENDIF()
