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

#ifndef TRACKING_DATA_TYPE
#define TRACKING_DATA_TYPE
#define MATRICES 0
#define QUATERNIONS 1
#endif

#ifndef TOLERANCE
#define TOLERANCE
#define TINY_NUMBER 0.000001
#endif

namespace niftk
{

typedef std::pair<mitk::Point4D,      // Rotation as Quaternion
                  mitk::Vector3D>     // Translation as Vector
                  RotationTranslation;

typedef std::pair<mitk::Image::Pointer,
                  RotationTranslation>
                  TrackedImage;

typedef std::vector<TrackedImage> TrackedImageData;

typedef std::pair<mitk::Point2D,
                  RotationTranslation>
                  TrackedPoint;

typedef std::vector<TrackedPoint> TrackedPointData;


/**
* \brief Loads data from 2 directories.
*/
NIFTKUSRECON_EXPORT TrackedImageData LoadImageAndTrackingDataFromDirectories(const std::string& imageDir,
                                                                             const std::string& trackingDir
                                                                            );

NIFTKUSRECON_EXPORT TrackedPointData MatchPointAndTrackingDataFromDirectories(const std::string& pointDir,
                                                                              const std::string& trackingDir  
                                                                             );

/**
* \brief Additional entry point for Guofang's Ultrasound Calibration,
* that takes images containing balls, extracts balls and calls DoUltrasoundPointCalibration.
*/
NIFTKUSRECON_EXPORT void DoUltrasoundBallCalibration(const int& ballSize,
                                                     const niftk::TrackedImageData& data,
                                                     mitk::Point2D& pixelScaleFactors,
                                                     niftk::RotationTranslation& imageToSensorTransform
                                                     );


/**
* \brief Main entry point for Guofang's Ultrasound Calibration.
* \param data pixel locations with matched tracking matrices.
*/
NIFTKUSRECON_EXPORT void DoUltrasoundPointCalibration(const niftk::TrackedPointData& data,
                                                      mitk::Point2D& pixelScaleFactors,
                                                      niftk::RotationTranslation& imageToSensorTransform
                                                     );


/**
* \brief Main entry point for Guofang's Ultrasound Reconstruction.
* \param imageToSensorTransform a pair containing rotation as quaternion and translation vector
* \param data images with matched tracking data as matrices.
*/
NIFTKUSRECON_EXPORT mitk::Image::Pointer DoUltrasoundReconstruction(const niftk::TrackedImageData& data,
                                                                    const mitk::Point2D& pixelScaleFactors,
                                                                    const niftk::RotationTranslation& imageToSensorTransform,
                                                                    const mitk::Vector3D& voxelSpacing
                                                                   );

} // end namespace

#endif
