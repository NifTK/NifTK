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

set(_exclude_plugins
  uk.ac.ucl.cmic.breastsegmentation
  uk.ac.ucl.cmic.gui.qt.niftyview
  uk.ac.ucl.cmic.gui.qt.niftymidas
  uk.ac.ucl.cmic.dnddisplay
  uk.ac.ucl.cmic.midasgeneralsegmentor
  uk.ac.ucl.cmic.midasmorphologicalsegmentor
  uk.ac.ucl.cmic.niftyseg
  uk.ac.ucl.cmic.thumbnail
  it.unito.cim.intensityprofile
  uk.ac.ucl.cmic.imagestatistics
  uk.ac.ucl.cmic.mitksegmentation
  uk.ac.ucl.cmic.niftyreg
  uk.ac.ucl.cmic.surfaceextractor
)

NIFTK_CREATE_GUI_APPLICATION(
  NAME NiftyIGI
  EXCLUDE_PLUGINS ${_exclude_plugins}
)
