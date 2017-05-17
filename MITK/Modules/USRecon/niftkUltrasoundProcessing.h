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
* \brief Loads and pairs images and tracking data from 2 directories.
* Called for calibration and reconstruction purposes.
* If the tracking data are in quaternion form, convert to matrices.
*/
NIFTKUSRECON_EXPORT MatrixTrackedImageData LoadImageAndTrackingDataFromDirectories(const std::string& imageDir,
                                                                             const std::string& trackingDir
                                                                            );

/**
* \brief Loads and pairs point files and tracking data from 2 directories.
* Called for calibration purpose.
* A point file contains the coordinates of a 3D point.
* Tracking data are in matrix form and are converted to quaternions.
*/
NIFTKUSRECON_EXPORT QuaternionTrackedPointData LoadPointAndTrackingDataFromDirectories(const std::string& pointDir,
                                                                              const std::string& trackingDir
                                                                             );

/**
* \brief Main entry point for Guofang's Ultrasound Calibration.
* Takes images containing balls, extracts balls and calls function UltrasoundCalibration.
* The diameter of the circle in the images should be measured with an interactive tool.
* Ring model width = diameter + 15
* Solvess scale factors in x and y directions, and hand-eye calibration in quaternion form.
* \param data ball images with matched tracking matrices.
*/
NIFTKUSRECON_EXPORT void DoUltrasoundBallCalibration(const int ballSize,
                                                     const niftk::MatrixTrackedImageData& data,
                                                     mitk::Point2D& pixelScaleFactors,
                                                     niftk::RotationTranslation& imageToSensorTransform
                                                     );


/**
* \brief Additional entry point for Guofang's Ultrasound Calibration.
* Calls function UltrasoundCalibration.
* Solves scale factors in x and y directions, and hand-eye calibration in quaternion form.
* \param data 2D point locations with matched tracking matrices.
*/
NIFTKUSRECON_EXPORT void DoUltrasoundPointCalibration(const niftk::QuaternionTrackedPointData& data,
                                                      mitk::Point2D& pixelScaleFactors,
                                                      niftk::RotationTranslation& imageToSensorTransform
                                                     );


/**
* \brief Main entry point for Guofang's 3D Free-hand Ultrasound Reconstruction.
* \param data images with matched tracking data as matrices.
* \param pixelScaleFactors scaling factor in x and y directions
* \param imageToSensorTransform a pair containing rotation as a quaternion and translation as a three-element vector
*/
NIFTKUSRECON_EXPORT mitk::Image::Pointer DoUltrasoundReconstruction(const niftk::MatrixTrackedImageData& data,
                                                                    const mitk::Point2D& pixelScaleFactors,
                                                                    const RotationTranslation& imageToSensorTransform,
                                                                    const mitk::Vector3D& voxelSpacing
                                                                   );

} // end namespace

#endif
