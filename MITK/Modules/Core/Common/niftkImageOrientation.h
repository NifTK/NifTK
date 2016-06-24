/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkImageOrientation_h
#define niftkImageOrientation_h

namespace niftk
{

/// \enum ImageOrientation
/// \brief Describes the different types of orientation, axial, sagittal, coronal,
/// that can be achieved in the Drag&Drop Display windows. This is different from
/// WindowOrientation. The orientation might be used to refer to the axis of an image,
/// so an image can ONLY be sampled in AXIAL, SAGITTAL and CORONAL direction.
enum ImageOrientation
{
  IMAGE_ORIENTATION_AXIAL = 0,
  IMAGE_ORIENTATION_SAGITTAL = 1,
  IMAGE_ORIENTATION_CORONAL = 2,
  IMAGE_ORIENTATION_UNKNOWN = 3
};

}

#endif
