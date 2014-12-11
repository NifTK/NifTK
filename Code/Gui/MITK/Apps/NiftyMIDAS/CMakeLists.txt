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

project(NiftyMIDAS)
set(MY_APP_NAME NiftyMIDAS)

###########################################################
# Note: You need to explicitly turn OFF MITK/CTK plugins.
#
# i.e. they are assumed to be well tested/safe.
# So, turn them OFF if you DON'T want them in your app.
###########################################################
set(_exclude_plugins
  org_mitk_gui_qt_extapplication
  org_mitk_gui_qt_volumevisualization
  org_mitk_gui_qt_pointsetinteraction
  org_mitk_gui_qt_stdmultiwidgeteditor
  org_mitk_gui_qt_segmentation
  org_mitk_gui_qt_measurementtoolbox
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
  uk.ac.ucl.cmic.gui.qt.commonmidas
  uk.ac.ucl.cmic.gui.qt.niftymidas
  uk.ac.ucl.cmic.gui.qt.common
  uk.ac.ucl.cmic.imagelookuptables
  uk.ac.ucl.cmic.snapshot
  uk.ac.ucl.cmic.imagestatistics
  uk.ac.ucl.cmic.xnat
  uk.ac.ucl.cmic.sideviewer
  uk.ac.ucl.cmic.thumbnail
  uk.ac.ucl.cmic.midasmorphologicalsegmentor
  uk.ac.ucl.cmic.midasgeneralsegmentor
  uk.ac.ucl.cmic.dnddisplay
)

NIFTK_CREATE_GUI_APPLICATION(
  NAME NiftyMIDAS
  INCLUDE_PLUGINS ${_include_plugins}
  EXCLUDE_PLUGINS ${_exclude_plugins}
)
