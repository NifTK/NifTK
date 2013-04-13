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

PROJECT(NiftyMIDAS)
SET(MY_APP_NAME NiftyMIDAS)

SET(_exclude_plugins
  org.mitk.gui.qt.pointsetinteraction
  org.mitk.gui.qt.stdmultiwidgeteditor
  org.mitk.gui.qt.volumevisualization
  it.unito.cim.intensityprofile 
  uk.ac.ucl.cmic.breastsegmentation
  uk.ac.ucl.cmic.gui.qt.niftyview
  uk.ac.ucl.cmic.gui.qt.niftyigi
  uk.ac.ucl.cmic.niftyseg
  uk.ac.ucl.cmic.surgicalguidance
  uk.ac.ucl.cmic.singlewidgeteditor
  uk.ac.ucl.cmic.igisurfacerecon
  uk.ac.ucl.cmic.igitagtracker
  uk.ac.ucl.cmic.igitrackedimage
)

NIFTK_CREATE_APPLICATION(
  NAME NiftyMIDAS
  EXCLUDE_PLUGINS ${_exclude_plugins}
)
