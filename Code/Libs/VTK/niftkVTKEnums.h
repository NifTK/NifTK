/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVTKEnums_h
#define niftkVTKEnums_h

/**
 * Compare this with enum SLICE_ORIENTATION_XY/XZ/YZ in vtkImageViewer2.
 * The one in vtkImageViewer2 is the data view, i.e. XY plane, XZ plane, YZ plane.
 * This one is the desired view as the user would expect to specify it.
 */
enum ViewerSliceOrientation
{
  VIEWER_ORIENTATION_CORONAL = 0,
  VIEWER_ORIENTATION_SAGITTAL = 1,
  VIEWER_ORIENTATION_AXIAL = 2,
  VIEWER_ORIENTATION_UNKNOWN = 3
};

#endif
