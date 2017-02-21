/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkUSReconstructor_h
#define niftkUSReconstructor_h

#include "niftkUSReconExports.h"
#include <mitkImage.h>
#include <niftkCoordinateAxesData.h>

namespace niftk
{
typedef std::pair<mitk::Image::Pointer, niftk::CoordinateAxesData::Pointer> TrackedImage;
typedef std::vector<TrackedImage> TrackedImageData;

/**
* \brief Entry point for Guofang's Ultrasound Calibration.
*/
NIFTKUSRECON_EXPORT std::vector<double> DoUltrasoundCalibration(const TrackedImageData& data);

/**
* \brief Entry point for Guofang's Ultrasound Reconstruction.
*/
NIFTKUSRECON_EXPORT mitk::Image::Pointer DoUltrasoundReconstruction(const TrackedImageData& data);


} // end namespace

#endif
