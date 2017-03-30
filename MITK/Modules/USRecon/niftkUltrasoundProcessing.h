/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkUltrasoundProcessing_h
#define niftkUltrasoundProcessing_h

#include "niftkUSReconExports.h"
#include <mitkImage.h>
#include <mitkVector.h>
#include <mitkPoint.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>


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
typedef std::pair<mitk::Image::Pointer,
                  vtkSmartPointer<vtkMatrix4x4>>
                  MatrixTrackedImage;

typedef std::vector<MatrixTrackedImage> MatrixTrackedImageData;

typedef std::pair<mitk::Point4D,      // Rotation as Quaternion
                  mitk::Vector3D>     // Translation as Vector
                  RotationTranslation;

typedef std::pair<mitk::Image::Pointer,
                  RotationTranslation>
                  QuaternionTrackedImage;

typedef std::vector<QuaternionTrackedImage> QuaternionTrackedImageData;

typedef std::pair<mitk::Point2D,
                  RotationTranslation>
                  QuaternionTrackedPoint;

typedef std::vector<QuaternionTrackedPoint> QuaternionTrackedPointData;


/**
* \brief Loads data from 2 directories.
*/
NIFTKUSRECON_EXPORT MatrixTrackedImageData LoadImageAndTrackingDataFromDirectories(const std::string& imageDir,
                                                                             const std::string& trackingDir
                                                                            );

NIFTKUSRECON_EXPORT QuaternionTrackedPointData LoadPointAndTrackingDataFromDirectories(const std::string& pointDir,
                                                                              const std::string& trackingDir
                                                                             );

/**
* \brief Main entry point for Guofang's Ultrasound Calibration.
* Takes images containing balls, extracts balls and calls DoUltrasoundPointCalibration.
* \param data ball images with matched tracking matrices.
*/
NIFTKUSRECON_EXPORT void DoUltrasoundBallCalibration(const int& ballSize,
                                                     const niftk::MatrixTrackedImageData& data,
                                                     mitk::Point2D& pixelScaleFactors,
                                                     niftk::RotationTranslation& imageToSensorTransform
                                                     );


/**
* \brief Additional entry point for Guofang's Ultrasound Calibration.
* \param data pixel locations with matched tracking data in quaternion form.
*/
NIFTKUSRECON_EXPORT void DoUltrasoundPointCalibration(const niftk::QuaternionTrackedPointData& data,
                                                      mitk::Point2D& pixelScaleFactors,
                                                      niftk::RotationTranslation& imageToSensorTransform
                                                     );


/**
* \brief Main entry point for Guofang's Ultrasound Reconstruction.
* \param imageToSensorTransform a pair containing rotation as quaternion and translation vector
* \param data images with matched tracking data as matrices.
*/
NIFTKUSRECON_EXPORT mitk::Image::Pointer DoUltrasoundReconstruction(const niftk::MatrixTrackedImageData& data,
                                                                    const mitk::Point2D& pixelScaleFactors,
                                                                    const RotationTranslation& imageToSensorTransform,
                                                                    const mitk::Vector3D& voxelSpacing
                                                                   );

} // end namespace

#endif
