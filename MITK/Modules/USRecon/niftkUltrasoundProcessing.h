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

namespace niftk
{

typedef std::pair<mitk::Image::Pointer, niftk::CoordinateAxesData::Pointer> TrackedImage;
typedef std::vector<TrackedImage> TrackedImageData;

/**
* \brief Entry point for Guofang's Ultrasound Calibration.
*/
NIFTKUSRECON_EXPORT void DoUltrasoundCalibration(const TrackedImageData& data,
                                                 vtkMatrix4x4& pixelToMillimetreScale,
                                                 vtkMatrix4x4& imageToSensorTransform
                                                );

cv::Point2d FindCircleInImage(const cv::Mat& image, cv::Mat& model);

cv::Mat CreateRingModel(const int model_width);

int HoughForRadius(cv::Mat& image, int x, int y, int& max_radius, int medianR);

int RawHough(cv::Mat& image, int& x, int& y, int medianR);

/**
* \brief Entry point for Guofang's Ultrasound Reconstruction.
*/
NIFTKUSRECON_EXPORT mitk::Image::Pointer DoUltrasoundReconstruction(const TrackedImageData& data,
                                                                    const vtkMatrix4x4& pixelToSensorTransform
                                                                   );

} // end namespace

#endif
