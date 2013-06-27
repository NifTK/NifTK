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

project(NiftyView)
set(MY_APP_NAME NiftyView)

set(_exclude_plugins
  uk.ac.ucl.cmic.breastsegmentation
  uk.ac.ucl.cmic.gui.qt.niftyigi
  uk.ac.ucl.cmic.gui.qt.niftymidas
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
  NAME NiftyView
  EXCLUDE_PLUGINS ${_exclude_plugins}
)
