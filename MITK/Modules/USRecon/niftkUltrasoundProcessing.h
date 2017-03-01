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
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <cv.h>
#include <niftkQuaternion.h>

namespace niftk
{

typedef std::pair<mitk::Image::Pointer, niftk::CoordinateAxesData::Pointer> TrackedImage;
typedef std::vector<TrackedImage> TrackedImageData;

//A pair of quaternions representing rotation and translation
typedef std::pair<niftkQuaternion, niftkQuaternion> TrackingQuaternions;

typedef std::pair<mitk::Image::Pointer, TrackingQuaternions> QuaternionTrackedImage;
typedef std::vector<QuaternionTrackedImage> QuaternionTrackedImageData;

/**
* \brief Entry point for Guofang's Ultrasound Calibration.
*/
int HoughForRadius(const cv::Mat& image, int x, int y, int& max_radius, int medianR);

void RawHough(const cv::Mat& image, int& x, int& y, int& r, int medianR);

cv::Mat CreateRingModel(const int model_width);

cv::Point2d FindCircleInImage(const cv::Mat& image, cv::Mat& model);

cv::Mat UltrasoundCalibration(const std::vector<cv::Point2d>& points,
                              const std::vector<TrackingQuaternions>& tracking_data);

NIFTKUSRECON_EXPORT void DoUltrasoundCalibration(const QuaternionTrackedImageData& data,
                                                 double& pixelToMillimetreScaleX,
                                                 double& pixelToMillimetreScaleY,
                                                 TrackingQuaternions& imageToSensorTransform
                                                );

//Calibration using transformation matrix - not implemented yet.
std::vector<double> UltrasoundCalibration(const std::vector<cv::Point2d>& points,
                                          const std::vector<cv::Matx44d>& matrices);

NIFTKUSRECON_EXPORT void DoUltrasoundCalibration(const TrackedImageData& data,
                                                 vtkMatrix4x4& pixelToMillimetreScale,
                                                 vtkMatrix4x4& imageToSensorTransform
                                                );


/**
* \brief Entry point for Guofang's Ultrasound Reconstruction.
*/
NIFTKUSRECON_EXPORT mitk::Image::Pointer DoUltrasoundReconstruction(const TrackedImageData& data,
                                                                    const vtkMatrix4x4& pixelToSensorTransform
                                                                   );

} // end namespace

#endif
