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
#include <mitkVector.h>
#include <mitkPoint.h>

namespace niftk
{

typedef std::pair<mitk::Point4D,      // Rotation as Quaternion
                  mitk::Vector3D>     // Translation as Vector
                  RotationTranslation;

typedef std::pair<mitk::Image::Pointer,
                  RotationTranslation>
                  TrackedImage;

typedef std::vector<TrackedImage> TrackedImageData;

/**
* \brief Entry point for Guofang's Ultrasound Calibration.
*/
NIFTKUSRECON_EXPORT void DoUltrasoundCalibration(const int& modelWidth,
                                                 const TrackedImageData& data,
                                                 mitk::Point2D& pixelScaleFactors,
                                                 RotationTranslation& imageToSensorTransform
                                                );


/**
* \brief Entry point for Guofang's Ultrasound Reconstruction.
*/
NIFTKUSRECON_EXPORT mitk::Image::Pointer DoUltrasoundReconstruction(const TrackedImageData& data,
                                                                    const mitk::Point2D& pixelScaleFactors,
                                                                    const RotationTranslation& imageToSensorTransform,
                                                                    const mitk::Vector3D& voxelSpacing
                                                                   );
} // end namespace

#endif
