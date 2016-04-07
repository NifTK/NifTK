Notes:
------

Compared with the original MITK version, this plugin is essentially:

1. Just the MITK standard editing tools "Add Subtract Paint Wipe 'Region Growing' Correction Fill Erase"
2. Only 2D mode, with no 2D or 3D spline based interpolation.

the reason for this is that our typical users did not want the interpolation mode, as it was confusing.

In addition the following changes have been made.

1. MITKSegmentation derives from QmitkMIDASBaseSegmentationFunctionality instead of QmitkFunctionality.
   This provides a different response to selection events, and makes it consistent across all our segmentation plugins.
   For all our segmentation plugins, you select the grey scale image and are given the "Create New" option.
   Thereafter, you select the binary image you want to segment.
2. QmitkSegmentationPreferencesPage: added a default colour preference, and removed export symbol.
3. QmitkCreatePolygonModelAction: removed export symbol.
4. Segmentation preferences node name changed to "/uk.ac.ucl.cmic.mitksegmentation"
