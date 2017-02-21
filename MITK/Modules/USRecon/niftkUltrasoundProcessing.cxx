/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "niftkUltrasoundProcessing.h"
#include <mitkExceptionMacro.h>

namespace niftk
{

//-----------------------------------------------------------------------------
mitk::Image::Pointer DoUltrasoundReconstruction(const TrackedImageData& data)
{
  MITK_INFO << "DoUltrasoundReconstruction: Doing Ultrasound Reconstruction with "
            << data.size() << " samples.";

  // This just creates a dummy image.
  // This should create a new image, and fill it with reconstructed data.
  mitk::PixelType pt = mitk::MakeScalarPixelType<unsigned char>();
  mitk::Image::Pointer op = mitk::Image::New();
  unsigned int dim[] = { 5, 5, 5 };
  op->Initialize( pt, 3, dim);

  // And returns the image.
  return op;
}


//-----------------------------------------------------------------------------
std::vector<double> DoUltrasoundCalibration(const TrackedImageData& data)
{
  MITK_INFO << "DoUltrasoundCalibration: Doing Ultrasound Calibration with "
            << data.size() << " samples.";

  std::vector<double> parameters;
  return parameters;
}

} // end namespace
