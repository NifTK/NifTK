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

###################################################################
# Plug-ins must be ordered according to their dependencies.
#
# Imagine that the build process, configures these plugins
# in order, from top to bottom. So, if plugin A depends on
# plugin B, then plugin B must be configured BEFORE plugin
# A. i.e. nearer the top of this file.
#
# This is of particular importance if you change a base
# class of a plugin. For example lets say your plugin
# currently depends on MITK's QmitkAbstractView, and
# hence you declare a dependency in manifest_headers.cmake
# on org.mitk.gui.qt.common.
#
# If you then subsequently change to derive from QmitkBaseView
# from the NifTK project, and declare a dependency in
# manifest_headers.cmake on uk.ac.ucl.cmic.gui.qt.common
# then your plugin must occur AFTER uk.ac.ucl.cmic.gui.qt.common.
#
# This is difficult to spot, as quite typically, during development
# then will be multiple build, config, build, config cycles
# anyway, such that the plugin will appear to be correctly
# configured. Worse than this, if you see a problem and then
# re-run cmake, the problem will disappear, but will re-appear
# on the next full clean build.
###################################################################

set(PROJECT_PLUGINS

# These are 'Application' plugins, and so are 'View' independent.
  Plugins/uk.ac.ucl.cmic.gui.qt.commonapps:ON
  Plugins/uk.ac.ucl.cmic.gui.qt.niftyview:ON

# These are 'View' plugins, and just depend on MITK.
  Plugins/uk.ac.ucl.cmic.snapshot:ON
  Plugins/uk.ac.ucl.cmic.thumbnail:ON    
  Plugins/uk.ac.ucl.cmic.imagestatistics:ON
  Plugins/uk.ac.ucl.cmic.midaseditor:ON      
  Plugins/uk.ac.ucl.cmic.xnat:OFF
  Plugins/uk.ac.ucl.cmic.niftyreg:ON                      # Must be after the xnat plugin
  Plugins/uk.ac.ucl.cmic.niftyseg:OFF                     # Not ready yet.
  Plugins/uk.ac.ucl.cmic.breastsegmentation:OFF           # Under development

# This 'common' plugin is our preferred base class for things that can't just derive from MITK.
  Plugins/uk.ac.ucl.cmic.gui.qt.common:ON
  Plugins/it.unito.cim.intensityprofile:OFF
  Plugins/uk.ac.ucl.cmic.imagelookuptables:ON
  Plugins/uk.ac.ucl.cmic.affinetransform:ON
  Plugins/uk.ac.ucl.cmic.surfaceextractor:ON

# Plugins listed after 'commonlegacy' depend on it, and the list below this plugin must be as short as possible.
  Plugins/uk.ac.ucl.cmic.gui.qt.commonlegacy:ON
)


# ---------------------------------------------------------------------------------------------------
# MIDAS Specific Plugins
# ---------------------------------------------------------------------------------------------------

if(BUILD_MIDAS)
  set(PROJECT_PLUGINS
    ${PROJECT_PLUGINS}
    Plugins/uk.ac.ucl.cmic.gui.qt.commonmidas:ON  
    Plugins/uk.ac.ucl.cmic.gui.qt.niftymidas:ON    
    Plugins/uk.ac.ucl.cmic.mitksegmentation:ON
    Plugins/uk.ac.ucl.cmic.midasmorphologicalsegmentor:ON
    Plugins/uk.ac.ucl.cmic.midasgeneralsegmentor:ON 
  )
endif(BUILD_MIDAS)


# ---------------------------------------------------------------------------------------------------
# IGI Specific Plugins
# ---------------------------------------------------------------------------------------------------

set(IGI_PLUGINS
  Plugins/uk.ac.ucl.cmic.gui.qt.niftyigi:ON
  Plugins/uk.ac.ucl.cmic.igioverlayeditor:ON
  Plugins/uk.ac.ucl.cmic.igidatasources:ON
  Plugins/uk.ac.ucl.cmic.igitagtracker:ON
  Plugins/uk.ac.ucl.cmic.igisurfacerecon:ON
  Plugins/uk.ac.ucl.cmic.igitrackedimage:ON
  Plugins/uk.ac.ucl.cmic.igitrackedpointer:ON
  Plugins/uk.ac.ucl.cmic.igipointreg:ON
  Plugins/uk.ac.ucl.cmic.igisurfacereg:ON
  Plugins/uk.ac.ucl.cmic.igiundistort:ON
)

if(BUILD_IGI)
  set(PROJECT_PLUGINS
    ${PROJECT_PLUGINS}
    ${IGI_PLUGINS}
  )
endif()
