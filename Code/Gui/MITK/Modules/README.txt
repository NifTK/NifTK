Notes:
------

The increasing number of Modules can be explained as follows:

1. niftMitkExt = represents extensions to Mitk. 
                 Named analagously to MitkExt. 
                 Also contains MIDAS stuff.
                 Does not contain GUI widgets.
                 
2. niftkQmitkExt = represents extension to Qmitk.
                   Named analagously to QmitkExt.
                   Also contains MIDAS stuff.
                   Contains GUI widgets.
                     
3. XnatRest = XNat API, independent of GUI widgets
4. XNatRest = XNat widgets.

5. niftkOpenCV = General purpose interface to OpenCV stuff. 
                 Only compiled if MITK and NiftyIGI are on.
                       
So, 1+2 and 3+4 demonstrate the principle that functionality, and GUI representation (widgets)
should be separated. This makes unit testing the main functionality easier.

For surgical guidance, we have the following:

niftkIGI = defines the data type, and a data source
niftkIGIGui = defines the GUI representation of a data source, depends on niftkIGI.
              However, as NiftyLink uses Qt, some of the data sources themselves ended up here.
              The GUI must be a separate class, named <data source>GUI.
niftkNVidia = depends on niftkIGI, just for the NVidia SDI stuff
niftkNVidiaGui = depends on niftkIGIGui and hence niftkIGI, for GUI controllers for the NVidia SDI stuff.

The reason the NVidia stuff is separate is that MOST platforms will NOT build this.

niftkIGIGuiManager = then has to be a separate module, with a dynamically configurable list of dependencies.

The uk_ac_ucl_cmic_gui_qt_surgicalguidance plugin then depends on niftkIGIGuiManager.                                                 