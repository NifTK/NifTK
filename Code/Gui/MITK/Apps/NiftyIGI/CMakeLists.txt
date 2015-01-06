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

project(NiftyIGI)
set(MY_APP_NAME NiftyIGI)

###########################################################
# Note: You need to explicitly turn OFF MITK/CTK plugins.
#
# i.e. they are assumed to be well tested/safe.
# So, turn them OFF if you DON'T want them in your app.
###########################################################
set(_exclude_plugins
  org_mitk_gui_qt_extapplication
  org_mitk_gui_qt_volumevisualization
  org_mitk_gui_qt_segmentation
  org_mitk_gui_qt_moviemaker
)

###########################################################
# Note: You need to explicitly turn ON NifTK plugins.
#
# i.e. YOU are explicitly stating (via the git commit log)
# that YOU have tested them on Windows, Linux and Mac,
# and are convinced that they work with this applications
# set of Editors and Views/Plugins.
###########################################################
set(_include_plugins
  uk.ac.ucl.cmic.gui.qt.commonapps
  uk.ac.ucl.cmic.gui.qt.niftyigi
  uk.ac.ucl.cmic.gui.qt.common
  uk.ac.ucl.cmic.affinetransform
  uk.ac.ucl.cmic.surfaceextractor
  uk.ac.ucl.cmic.snapshot
  uk.ac.ucl.cmic.igioverlayeditor
  uk.ac.ucl.cmic.igidatasources
  uk.ac.ucl.cmic.igisurfacerecon
  uk.ac.ucl.cmic.igitrackedimage
  uk.ac.ucl.cmic.igitrackedpointer
  uk.ac.ucl.cmic.igipointreg
  uk.ac.ucl.cmic.igisurfacereg
  uk.ac.ucl.cmic.igiundistort
  uk.ac.ucl.cmic.igirmserror
)

if(BUILD_VL)
  set(_include_plugins
    ${_include_plugins}
    uk.ac.ucl.cmic.newvisualization
  )
endif()

NIFTK_CREATE_GUI_APPLICATION(
  NAME NiftyIGI
  INCLUDE_PLUGINS ${_include_plugins}
  EXCLUDE_PLUGINS ${_exclude_plugins}
)
