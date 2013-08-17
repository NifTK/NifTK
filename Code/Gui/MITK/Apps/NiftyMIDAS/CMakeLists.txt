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

set(_exclude_plugins
  org.mitk.gui.qt.common.legacy
  org.mitk.gui.qt.pointsetinteraction
  org.mitk.gui.qt.stdmultiwidgeteditor
  org.mitk.gui.qt.volumevisualization
  it.unito.cim.intensityprofile
  uk.ac.ucl.cmic.breastsegmentation
  uk.ac.ucl.cmic.gui.qt.commonlegacy
  uk.ac.ucl.cmic.gui.qt.niftyview
  uk.ac.ucl.cmic.gui.qt.niftyigi
  uk.ac.ucl.cmic.niftyseg
  uk.ac.ucl.cmic.igidatasources
  uk.ac.ucl.cmic.igioverlayeditor
  uk.ac.ucl.cmic.igipointreg
  uk.ac.ucl.cmic.igisurfacerecon
  uk.ac.ucl.cmic.igisurfacereg
  uk.ac.ucl.cmic.igitagtracker
  uk.ac.ucl.cmic.igitrackedimage
  uk.ac.ucl.cmic.igitrackedpointer
  uk.ac.ucl.cmic.igiundistort
)

NIFTK_CREATE_GUI_APPLICATION(
  NAME NiftyMIDAS
  EXCLUDE_PLUGINS ${_exclude_plugins}
)
