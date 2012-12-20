
# Plug-ins must be ordered according to their dependencies

set(PROJECT_PLUGINS

# These are 'Application' plugins, and so are 'View' independent.
  Plugins/uk.ac.ucl.cmic.gui.qt.commonapps:ON 
  Plugins/uk.ac.ucl.cmic.gui.qt.niftyview:ON
  Plugins/uk.ac.ucl.cmic.gui.qt.niftyigi:ON
  Plugins/uk.ac.ucl.cmic.gui.qt.niftymidas:ON
  
# These are 'View' plugins, and just depend on MITK.
  Plugins/uk.ac.ucl.cmic.imagelookuptables:ON
  Plugins/uk.ac.ucl.cmic.snapshot:ON
  Plugins/uk.ac.ucl.cmic.thumbnail:ON
  Plugins/uk.ac.ucl.cmic.imagestatistics:ON
  Plugins/uk.ac.ucl.cmic.surfaceextractor:ON
  Plugins/uk.ac.ucl.cmic.midaseditor:ON
  Plugins/uk.ac.ucl.cmic.xnat:ON
  Plugins/uk.ac.ucl.cmic.niftyreg:ON                      # Must be after the xnat plugin
  Plugins/uk.ac.ucl.cmic.niftyseg:OFF                     # Not ready yet.
  Plugins/uk.ac.ucl.cmic.breastsegmentation:OFF           # Under development
  Plugins/uk.ac.ucl.cmic.singlewidgeteditor:ON           # Under development
 
# This 'common' plugin is our preferred base class for things that can't just derive from MITK.  
  Plugins/uk.ac.ucl.cmic.gui.qt.common:ON
  Plugins/uk.ac.ucl.cmic.affinetransform:ON
  Plugins/it.unito.cim.intensityprofile:ON
  
# This 'commonmidas' depends on 'common' and serves like 'base classes' for MIDAS segmentation stuff.  
  Plugins/uk.ac.ucl.cmic.gui.qt.commonmidas:ON            
  Plugins/uk.ac.ucl.cmic.mitksegmentation:ON  
  Plugins/uk.ac.ucl.cmic.midasmorphologicalsegmentor:ON
  Plugins/uk.ac.ucl.cmic.midasgeneralsegmentor:ON
  
# Plugins listed after 'commonlegacy' depend on it, and this list must be as short as possible.
  Plugins/uk.ac.ucl.cmic.gui.qt.commonlegacy:ON           
  Plugins/uk.ac.ucl.cmic.surgicalguidance:ON  
)
