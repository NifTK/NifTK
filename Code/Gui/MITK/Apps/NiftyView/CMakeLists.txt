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

###########################################################
# Note: You need to explicitly turn OFF MITK/CTK plugins.
#
# i.e. they are assumed to be well tested/safe.
# So, turn them OFF if you DON'T want them in your app.
###########################################################
set(_exclude_plugins
  org_mitk_gui_qt_extapplication
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
  uk.ac.ucl.cmic.gui.qt.niftyview
  uk.ac.ucl.cmic.gui.qt.common
  uk.ac.ucl.cmic.imagelookuptables
  uk.ac.ucl.cmic.affinetransform
  uk.ac.ucl.cmic.surfaceextractor
  uk.ac.ucl.cmic.snapshot
  uk.ac.ucl.cmic.imagestatistics
  uk.ac.ucl.cmic.xnat
)


if(BUILD_VL)
  set(_include_plugins
    ${_include_plugins}
    uk.ac.ucl.cmic.vlrenderer
  )
endif()

NIFTK_CREATE_GUI_APPLICATION(
  NAME NiftyView
  INCLUDE_PLUGINS ${_include_plugins}
  EXCLUDE_PLUGINS ${_exclude_plugins}
)
