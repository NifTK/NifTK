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
# MITK
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED MITK_DIR AND NOT EXISTS ${MITK_DIR})
  MESSAGE(FATAL_ERROR "MITK_DIR variable is defined but corresponds to non-existing directory \"${MITK_DIR}\".")
ENDIF()

SET(proj MITK)
SET(proj_DEPENDENCIES BOOST ITK VTK GDCM DCMTK)  # Don't put CTK here, as it's optional, dependent on Qt.
IF(QT_FOUND)
  SET(proj_DEPENDENCIES BOOST ITK VTK GDCM DCMTK CTK)
ENDIF(QT_FOUND)
SET(MITK_DEPENDS ${proj})

IF(NOT DEFINED MITK_DIR)

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
      set(MITK_BUILD_org.mitk.gui.qt.pointsetinteraction ON CACHE BOOL \"Build the MITK point set interaction plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.volumevisualization ON CACHE BOOL \"Build the MITK volume visualization plugin\")      
      set(MITK_BUILD_org.mitk.gui.qt.stdmultiwidgeteditor ON CACHE BOOL \"Build the MITK ortho-viewer plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.segmentation OFF CACHE BOOL \"Build the MITK segmentation plugin\")
      set(MITK_BUILD_org.mitk.gui.qt.cmdlinemodules ON CACHE BOOL \"Build the Command Line Modules plugin. \")
      set(BLUEBERRY_BUILD_org.blueberry.ui.qt.log ON CACHE BOOL \"Build the Blueberry logging plugin\")
      set(BLUEBERRY_BUILD_org.blueberry.ui.qt.help ON CACHE BOOL \"Build the Blueberry Qt help plugin\")
      set(BLUEBERRY_BUILD_org.blueberry.compat ON CACHE BOOL \"Build the Blueberry compat plugin (Matt, what is this for?)\")
      set(BOOST_INCLUDEDIR ${BOOST_INCLUDEDIR} CACHE PATH \"Path to Boost include directory\")
      set(BOOST_LIBRARYDIR ${BOOST_LIBRARYDIR} CACHE PATH \"Path to Boost library directory\")
      set(DCMTK_DIR ${DCMTK_DIR} CACHE PATH \"Path to DCMTK installation directory\")
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
    # 4. Trac 1429 - merged MITK master 58f3a08c84 (08.05.2012 08:46) onto niftk branch
    #    to pick up MITK bug 11820. So the current build is effectively MITK master 
    #    58f3a08c84 plus the trac items 853, 1256, 1344 and 1366 listed above. The version
    #    we are using is thus 1bbf9184dd on MattClarkson github fork.
    # 
    # 5. Trac 1429 - merged MITK master 4ddb84dc4e (22.05.2012 08:09) onto niftk branch
    #    to pick up MITK latest, including such fixes as 11913 for apple. So the current
    #    build is effectively MITK master 4ddb84dc4e plus the trac items 853, 1256, 
    #    1344 and 1366 listed above. The version we are using is thus 1da33a0b08 on
    #    MattClarkson github fork.
    #
    # 6. Trac 1482 - merged MITK master 53aba30c0c (Sat May 26 14:58:13 2012 +0200) onto niftk
    #    branch, to pick up MITK latest. Then created branches 
    #    MITK-bug-10420-trac-1479-render-window-steal-mouse-clicks
    #    MITK-bug-12002-trac-1467-QmitkFunctionality-crash-if-QmitkStdMultiWidgetEditor-not-the-only-editor
    #    MITK-bug-12003-trac-1469-Make-crosses-only-appear-on-current-slice
    #    locally, then merged into niftk, to push niftk version fdfefc50c9.
    #    Thus, the effective codebase is:
    #    MITK version 53aba30c0c plus:
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/853  (opacity for black. MITK working on proper fix).
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1256 (improve file extension gz. Not entirely merged ... needs re-checking).
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1344 (Turn off interactors in mitkMouseModeSwitcher)
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1366 (Don't create PACS interactors in mitkMouseModeSwitcher)
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1467 (Stop QmitkFunctionality crash when QmitkStdMultiWidgetEditor not the only editor)
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1469 (Make crosses only appear on current slice)
    #    https://cmicdev.cs.ucl.ac.uk/trac/ticket/1479 (Stop QmitkRenderWindow steal mouse clicks).
    #
    # 7. Trac 1482 - updated MITK-bug-10420-trac-1479-render-window-steal-mouse-clicks, and merged back onto niftk branch.
    #    This produced commit number b716de0d73.
    #
    # 8. Trac 1494 - Merged MITK commit 250107a06aa onto niftk to produce 1ad3934a4a.
    #    MITK had also merged our trac changes 1467 (MITK 12002) and 1479 (MITK bug 10420), so the 
    #    current code base would be MITK 250107a06aa + Trac 853, 1256, 1344, 1366, 1469.
    #
    # 9. Trac 1494 - Merged MITK commit db2f383b66 onto niftk to produce efe1377731.
    #    MITK also merged our trac changes 1497 (MITK bug 12084), so the current code base is
    #    MITK db2f383b66 + Trac 853, 1256, 1344, 1366, 1469, which is efe1377731 on niftk branch. 
    #
    # 10. Trac 1585 - Merged MITK commit 3d35ed8ff3 onto niftk. This includes DICOM fixes for Trac 1456, 1472 and 1575.
    #                 Merged MITK bug 12302, 12303 from Matt Clarkson github, straight into niftk branch.
    #     So the current code base is:
    #       MITK 3d35ed8ff3 + Trac 853, 1256, + MITK 12302, 12303 (not yet merged into MITK master).
    #       This results in niftk branch commit d1879853b1   
    #                 
    # 11. Trac 1600 - Merged MITK commit c855dedda4 onto niftk due to pickup fix for MITK bug 12299.
    #     So, the current code base is:
    #       MITK c855dedda4 + Trac 853, 1256, + MITK 12302, 12303 (not yet merged into MITK master).
    #       This results in niftk branch commit f6bbb3c5f4
    #
    # 12. Trac 1641 - Created MITK bugs 12427 to make sure mitkDicomSeriesReader compiles on gcc 4.1.2,
    #     and also MITK bug 12431 to fix crash in mitkExtractSliceFilter. Merged these straight into niftk branch.
    #     So, the current code base is:
    #       MITK c855dedda4 + Trac 853, 1256, + MITK 12302, 12303, 12427 and 12431 (not yet merged into MITK master).
    #       This results in niftk brach commit 6f6ff4eeb2
    #
    # 13. Trac 1757 - MITK upgrade
    #     A minor change is committed to the branch for MITK 12302 and merged back into the niftk branch.
    #     It was needed because of a change in the CTK API.
    #       This results in niftk branch commit 9df515e9ef
    #  
    # 14. Trac 1757 - New MITK version to pick up latest changes as we head for 12.09 release.
    #               - MITK bug 12427 now in MITK master
    #               - MITK bug 12302 changes mean that our changes to turn interactors off/in need to be backed out
    #                 and re-worked on the NifTK side, as the whole interaction pattern has changed.
    #
    #     Current MITK code base (i.e. if we had to recreate from scratch) is in effect:
    #       MITK d2581aea00 - Sep 14 2012
    #       + Trac 853,  MITK 10174 = https://github.com/MattClarkson/MITK/commit/5d11b54efc00cd8ddf086b2c6cbac5f6a6eae315 (Opacity for black)
    #       + Trac 1256, MITK 10783 = https://github.com/MattClarkson/MITK/commit/82efd288c7f7b5b5d098e33e2de6fc83c8ed79b7 (gz file extension handling)
    #       + Trac 1584, MITK 12303 = https://github.com/MattClarkson/MITK/commit/c9f7b430ea615efe0303afa37824d276486eb442 (Axial instead of Transversal)
    #       + Trac 1628, MITK 12431 = https://github.com/MattClarkson/MITK/commit/3976cb339ba7468815ffbf96f85bd36b832aa648 (Dont crash if bounding box invalid)
    #       + Trac 1469, MITK 12003 = https://github.com/MattClarkson/MITK/commit/6dc50f81de6ad7b9c3344554d0a4dc53867112f9 (Crosses not on out of plane slices)
    #       + Trac 1781, MITK 13113 = https://github.com/MattClarkson/MITK/commit/598ee13b691224cb07fa89bc264271a96e6e35ce (Reintroduce SegTool2D::SetEnable3DInterpolation)
    #
    #     Giving 5d26e4b046 on the niftk branch
    #
    # 15. Trac 1588 - Merge in MITK plugin for Slicer Command Line Modules
    #     
    #     Current MITK code base (i.e. if we had to recreate from scratch) is in effect:
    #       MITK d2581aea00 - Sep 14 2012
    #       + Trac 853,  MITK 10174 = https://github.com/MattClarkson/MITK/commit/5d11b54efc00cd8ddf086b2c6cbac5f6a6eae315 (Opacity for black)
    #       + Trac 1256, MITK 10783 = https://github.com/MattClarkson/MITK/commit/82efd288c7f7b5b5d098e33e2de6fc83c8ed79b7 (gz file extension handling)
    #       + Trac 1584, MITK 12303 = https://github.com/MattClarkson/MITK/commit/c9f7b430ea615efe0303afa37824d276486eb442 (Axial instead of Transversal)
    #       + Trac 1628, MITK 12431 = https://github.com/MattClarkson/MITK/commit/3976cb339ba7468815ffbf96f85bd36b832aa648 (Dont crash if bounding box invalid)
    #       + Trac 1469, MITK 12003 = https://github.com/MattClarkson/MITK/commit/6dc50f81de6ad7b9c3344554d0a4dc53867112f9 (Crosses not on out of plane slices)
    #       + Trac 1781, MITK 13113 = https://github.com/MattClarkson/MITK/commit/598ee13b691224cb07fa89bc264271a96e6e35ce (Reintroduce SegTool2D::SetEnable3DInterpolation)
    #       + Trac 1588, MITK 12506 = https://github.com/MattClarkson/MITK/commit/576f66720701045b914a3337870353744268094f (Slicer Command Line Modules)
    #
    #     Giving db37592aa3 on the niftk branch
    #
    # 16. Trac 1784 - Merge latest MITK 2012.09.0 release and latest Slicer Command Line Module work, plus new bugfix 11627.
    #               - MITK 12303 is now in MITK master
    #               - MITK 13113 is now in MITK master, but it is after the release hashtag below, so will disappear from this list at the next update.
    #
    #     Current MITK code base (i.e. if we had to recreate from scratch) is in effect:
    #       MITK b6cfb353a9 - Sep 19 2012 = 2012.09.0 release
    #       + Trac 853,  MITK 10174 = https://github.com/MattClarkson/MITK/commit/5d11b54efc00cd8ddf086b2c6cbac5f6a6eae315 (Opacity for black)
    #       + Trac 1256, MITK 10783 = https://github.com/MattClarkson/MITK/commit/82efd288c7f7b5b5d098e33e2de6fc83c8ed79b7 (gz file extension handling)
    #       + Trac 1628, MITK 12431 = https://github.com/MattClarkson/MITK/commit/3976cb339ba7468815ffbf96f85bd36b832aa648 (Dont crash if bounding box invalid)
    #       + Trac 1469, MITK 12003 = https://github.com/MattClarkson/MITK/commit/6dc50f81de6ad7b9c3344554d0a4dc53867112f9 (Crosses not on out of plane slices)
    #       + Trac 1781, MITK 13113 = https://github.com/MattClarkson/MITK/commit/598ee13b691224cb07fa89bc264271a96e6e35ce (Reintroduce SegTool2D::SetEnable3DInterpolation)
    #       + Trac 1588, MITK 12506 = https://github.com/MattClarkson/MITK/commit/576f66720701045b914a3337870353744268094f (Slicer Command Line Modules)
    #       + Trac 1791, MITK 11627 = https://github.com/MattClarkson/MITK/commit/0196305455913856beb251dd58e69df3e6a86e37 (Fix Analyze file name)
    # 
    #     Giving 875bde5a2b on the niftk branch
    #
    # 17. Trac 1821 - Merge Latest Slicer Command Line Module work from:
    #       https://github.com/MattClarkson/MITK/commit/6bca0b2907b374aabbb5a6110ac6a2f7a06ad8b0
    #     Results in change to niftk branch, with no other MITK change.
    #
    #     Current MITK code base (i.e. if we had to recreate from scratch) is in effect:
    #       MITK b6cfb353a9 - Sep 19 2012 = 2012.09.0 release
    #       + Trac 853,  MITK 10174 = https://github.com/MattClarkson/MITK/commit/5d11b54efc00cd8ddf086b2c6cbac5f6a6eae315 (Opacity for black)
    #       + Trac 1256, MITK 10783 = https://github.com/MattClarkson/MITK/commit/82efd288c7f7b5b5d098e33e2de6fc83c8ed79b7 (gz file extension handling)
    #       + Trac 1628, MITK 12431 = https://github.com/MattClarkson/MITK/commit/3976cb339ba7468815ffbf96f85bd36b832aa648 (Dont crash if bounding box invalid)
    #       + Trac 1469, MITK 12003 = https://github.com/MattClarkson/MITK/commit/6dc50f81de6ad7b9c3344554d0a4dc53867112f9 (Crosses not on out of plane slices)
    #       + Trac 1781, MITK 13113 = https://github.com/MattClarkson/MITK/commit/598ee13b691224cb07fa89bc264271a96e6e35ce (Reintroduce SegTool2D::SetEnable3DInterpolation)
    #       + Trac 1588, MITK 12506 = https://github.com/MattClarkson/MITK/commit/6bca0b2907b374aabbb5a6110ac6a2f7a06ad8b0 (Slicer Command Line Modules)
    #       + Trac 1791, MITK 11627 = https://github.com/MattClarkson/MITK/commit/0196305455913856beb251dd58e69df3e6a86e37 (Fix Analyze file name)   
    #
    # 18. Trac 1872 - Creating a new MITK version.
    #
    #     HOWEVER: With the MITK on NifTK/MITK/master latest, we merged MITK d70faf53e4 - Oct 26 2012 - 16:10:44
    #              and found that the MIDAS morph editor does not work due to changes to image accessors due to MITK
    #              bug: http://bugs.mitk.org/show_bug.cgi?id=13230
    #     
    #     SO: I took the latest niftk branch - commit 6bca0b2907 and cherry-picked:
    #
    #     Trac 1871, MITK 13504 = https://github.com/NifTK/MITK/commit/c874a341335812cf4c38b5c5daea4db4f4444c0d (CTK Designer plugin deployment)
    #                MITK 13495 = https://github.com/MITK/MITK/commit/d9a3bfade7e349d19fd06ae06ed4899a5bdd8a77  (dicom. fix already on MITK master)
    #
    #     Current MITK code base (i.e. if we have to recreate from scratch) is in effect:
    #       MITK b6cfb353a9 - Sep 19 2012 = 2012.09.0 release
    #       + Trac 853,  MITK 10174 = https://github.com/NifTK/MITK/commit/5d11b54efc00cd8ddf086b2c6cbac5f6a6eae315 (Opacity for black)
    #       + Trac 1256, MITK 10783 = https://github.com/NifTK/MITK/commit/82efd288c7f7b5b5d098e33e2de6fc83c8ed79b7 (gz file extension handling)
    #       + Trac 1628, MITK 12431 = https://github.com/NifTK/MITK/commit/3976cb339ba7468815ffbf96f85bd36b832aa648 (Dont crash if bounding box invalid)
    #       + Trac 1469, MITK 12003 = https://github.com/NifTK/MITK/commit/6dc50f81de6ad7b9c3344554d0a4dc53867112f9 (Crosses not on out of plane slices)
    #       + Trac 1781, MITK 13113 = https://github.com/NifTK/MITK/commit/598ee13b691224cb07fa89bc264271a96e6e35ce (Reintroduce SegTool2D::SetEnable3DInterpolation)
    #       + Trac 1588, MITK 12506 = https://github.com/NifTK/MITK/commit/6bca0b2907b374aabbb5a6110ac6a2f7a06ad8b0 (Slicer Command Line Modules)
    #       + Trac 1791, MITK 11627 = https://github.com/NifTK/MITK/commit/0196305455913856beb251dd58e69df3e6a86e37 (Fix Analyze file name)   
    #       + Trac 1871, MITK 13504 = https://github.com/NifTK/MITK/commit/c874a341335812cf4c38b5c5daea4db4f4444c0d (CTK Designer plugin deployment)
    #       +            MITK 13495 = https://github.com/MITK/MITK/commit/d9a3bfade7e349d19fd06ae06ed4899a5bdd8a77  (dicom. fix already on MITK master)
    # 
    #     Giving c3214181308907c12c2d62f6cb775da04411772d on NifTK/MITK/niftk-tmp-12.10 branch
    #
    # 19. Trac 1893 - New MITK and CTK version.
    #  
    #     Took the niftk branch - commit 6bca0b2907 and cherry-picked:
    #      
    #     Trac 1871, MITK 13504 = https://github.com/NifTK/MITK/commit/c874a341335812cf4c38b5c5daea4db4f4444c0d (CTK Designer plugin deployment)
    #                MITK 13495 = https://github.com/MITK/MITK/commit/d9a3bfade7e349d19fd06ae06ed4899a5bdd8a77  (dicom. fix already on MITK master)
    #                MITK 13386 = https://github.com/MITK/MITK/commit/f69ab59f0fa0a1df848e17f4d1d25c4ebcbdb0c7  (missing newline Geometry3D already on MITK master)
    #     Trac 1588, MITK 12506 = https://github.com/NifTK/MITK/commit/acffcb4f1f3a483026b891ae49f45688d597cff8 (latest Slicer Command Line Modules)
    #
    #     Current MITK code base (i.e. if we have to recreate from scratch) is in effect:
    #       MITK b6cfb353a9 - Sep 19 2012 = 2012.09.0 release
    #
    #       + Trac 853,  MITK 10174 = https://github.com/NifTK/MITK/commit/5d11b54efc00cd8ddf086b2c6cbac5f6a6eae315 (Opacity for black)
    #       + Trac 1256, MITK 10783 = https://github.com/NifTK/MITK/commit/82efd288c7f7b5b5d098e33e2de6fc83c8ed79b7 (gz file extension handling)
    #       + Trac 1628, MITK 12431 = https://github.com/NifTK/MITK/commit/3976cb339ba7468815ffbf96f85bd36b832aa648 (Dont crash if bounding box invalid)
    #       + Trac 1469, MITK 12003 = https://github.com/NifTK/MITK/commit/6dc50f81de6ad7b9c3344554d0a4dc53867112f9 (Crosses not on out of plane slices)
    #       + Trac 1781, MITK 13113 = https://github.com/NifTK/MITK/commit/598ee13b691224cb07fa89bc264271a96e6e35ce (Reintroduce SegTool2D::SetEnable3DInterpolation)
    #       + Trac 1791, MITK 11627 = https://github.com/NifTK/MITK/commit/0196305455913856beb251dd58e69df3e6a86e37 (Fix Analyze file name)   
    #       + Trac 1871, MITK 13504 = https://github.com/NifTK/MITK/commit/c874a341335812cf4c38b5c5daea4db4f4444c0d (CTK Designer plugin deployment)
    #       +            MITK 13495 = https://github.com/MITK/MITK/commit/d9a3bfade7e349d19fd06ae06ed4899a5bdd8a77  (dicom. fix already on MITK master)
    #       +            MITK 13386 = https://github.com/MITK/MITK/commit/f69ab59f0fa0a1df848e17f4d1d25c4ebcbdb0c7  (missing newline Geometry3D already on MITK master)
    #       + Trac 1588, MITK 12506 = https://github.com/NifTK/MITK/commit/acffcb4f1f3a483026b891ae49f45688d597cff8 (Slicer Command Line Modules)
    #
    #     Giving acffcb4f1f3a483026b891ae49f45688d597cff8 on NifTK/MITK/niftk-tmp-12.10 branch  
    #########################################################

    niftkMacroGetChecksum(NIFTK_CHECKSUM_MITK ${NIFTK_LOCATION_MITK})

    ExternalProject_Add(${proj}
      URL ${NIFTK_LOCATION_MITK}
      URL_MD5 ${NIFTK_CHECKSUM_MITK}
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
        -DMITK_USE_OpenCV:BOOL=${NIFTK_USE_OPENCV}
        -DADDITIONAL_C_FLAGS:STRING=${NIFTK_ADDITIONAL_C_FLAGS}
        -DADDITIONAL_CXX_FLAGS:STRING=${NIFTK_ADDITIONAL_CXX_FLAGS}
        -DBOOST_ROOT:PATH=${BOOST_ROOT}                        # FindBoost expectes BOOST_ROOT  
        -DBOOST_INCLUDEDIR:PATH=${BOOST_INCLUDEDIR}            # Derived from BOOST_ROOT, set in BOOST.cmake
        -DBOOST_LIBRARYDIR:PATH=${BOOST_LIBRARYDIR}            # Derived from BOOST_ROOT, set in BOOST.cmake
        -DGDCM_DIR:PATH=${GDCM_DIR}                            # FindGDCM expects GDCM_DIR
        -DVTK_DIR:PATH=${VTK_DIR}                              # FindVTK expects VTK_DIR
        -DITK_DIR:PATH=${ITK_DIR}                              # FindITK expects ITK_DIR
        -DCTK_DIR:PATH=${CTK_DIR}                              # FindCTK expects CTK_DIR
        -DDCMTK_DIR:PATH=${DCMTK_DIR}                          # FindDCMTK expects DCMTK_DIR
        -DMITK_INITIAL_CACHE_FILE:FILEPATH=${MITK_INITIAL_CACHE_FILE}
      DEPENDS ${proj_DEPENDENCIES}
      )
    SET(MITK_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build/${proj}-build)
    MESSAGE("SuperBuild loading MITK from ${MITK_DIR}")

ELSE()

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

ENDIF()
