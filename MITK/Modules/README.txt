Notes:
------

The increasing number of Modules can be explained as follows:

1. niftCore      = represents extensions to Mitk.
                   Does not contain GUI widgets.
                   Folder structure matches MITK.

2. niftkCoreGui  = represents extension to Qmitk.
                   Contains GUI widgets.
                   Folder structure is minimal with nearly all widgets at one level.

3. niftkMIDAS    = represents core functionality for MIDAS project.
                   Does not contain GUI widgets.
                   Folder structure matches MITK.

4. niftkMIDASGui = GUI widgets for MIDAS project.
                   Folder structure is minimal with nearly all widgets at one level.

5. niftkNVidia   = Non-GUI classes, for integrating NVidia SDI stuff into MITK app.

6. niftkNVidiGui = GUI classes, for integrating NVidia SDI stuff into MITK app.

7. niftkIGI      = Non-GUI classes, for general purpose IGI platform.

8. niftkIGIGui  = GUI classes, for general purpose IGI platform.

9. niftkOpenCV  = General purpose interface to OpenCV stuff.
                   Only compiled if MITK and NiftyIGI are on.

10. niftkIGIGuiManager = separate module, with a dynamically configurable list of dependencies.

So, each pair demonstrate the principle that functionality, and GUI representation (widgets)
should be separated. This makes unit testing the main functionality easier.

The reason the NVidia stuff is separate is that MOST platforms will NOT build this.
