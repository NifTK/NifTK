/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-06-01 09:38:00 +0100 (Wed, 01 Jun 2011) $
 Revision          : $Revision: 6322 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef VTKENUMS_H
#define VTKENUMS_H

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
