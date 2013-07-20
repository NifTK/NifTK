/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkTagTrackingFacade_h
#define mitkTagTrackingFacade_h

#include <cv.h>
#include <cstdlib>
#include <iostream>
#include <vtkMatrix4x4.h>
#include <mitkVector.h>
#include <mitkVector.h>
#include <itkPoint.h>
#include <mitkPointSet.h>

namespace mitk
{
  typedef itk::Point<mitk::ScalarType,6> Point6D;

/**
 * \brief Detect the AR-Tag style markers in distortion-corrected colour RGB images.
 * \param inImage colour RGB image.
 * \param minSize set the minimum size of the marker, as a fraction of the maximum(number rows, number columns).
 * \param maxSize set the maximum size of the marker, as a fraction of the maximum(number rows, number columns).
 * \param blockSize block size for adaptive thresholding
 * \param offset amount below mean for adaptive thresholding
 * \param drawOutlines if true will write on the inImage and outline the detected markers in red.
 * \param drawCentre if true will write on the inImage and draw the centre of the marker in blue.
 * \return map of marker id and its 2D pixel location of the centre of the marker.
 */
std::map<int, cv::Point2f> DetectMarkers(
    cv::Mat& inImage,
    const float& minSize = 0.01,
    const float& maxSize = 0.125,
    const double& blockSize = 7,
    const double& offset = 7,
    const bool& drawOutlines = false,
    const bool& drawCentre = false
    );


/**
 * \brief Detect the AR-Tag style markers in stereo distortion-corrected colour RGB images.
 * \param inImageLeft colour RGB image.
 * \param inImageRight colour RGB image.
 * \param intrinsicParamsLeft matrix of camera intrinsic parameters \see mitk::CameraCalibrationFacade.
 * \param intrinsicParamsRight matrix of camera intrinsic parameters \see mitk::CameraCalibrationFacade.
 * \param rightToLeftRotationVector [1x3] vector representing the rotation between camera axes
 * \param rightToLeftTranslationVector [1x3] translation between camera origins
 * \param minSize set the minimum size of the marker, as a fraction of the maximum(number rows, number columns).
 * \param maxSize set the maximum size of the marker, as a fraction of the maximum(number rows, number columns).
 * \param blockSize block size for adaptive thresholding
 * \param offset amount below mean for adaptive thresholding
 * \param drawOutlines if true will write on the inImage and outline the detected markers in red.
 * \param drawCentre if true will write on the inImage and draw the centre of the marker in blue.
 * \return map of marker id and its 3D pixel location.
 */
std::map<int, cv::Point3f> DetectMarkerPairs(
    cv::Mat& inImageLeft,
    cv::Mat& inImageRight,
    const cv::Mat& intrinsicParamsLeft,
    const cv::Mat& intrinsicParamsRight,
    const cv::Mat& rightToLeftRotationVector,
    const cv::Mat& rightToLeftTranslationVector,
    const float& minSize = 0.01,
    const float& maxSize = 0.125,
    const double& blockSize = 7,
    const double& offset = 7,
    const bool& drawOutlines = false,
    const bool& drawCentre = false
    );


/**
 * \brief Highly Experimental method to extract marker pairs, and compute surface normals for each extracted point.
 * \return map of marker id and a 6D point containing the marker centre and its surface normal.
 */
std::map<int, Point6D> DetectMarkerPairsAndNormals(
    cv::Mat& inImageLeft,
    cv::Mat& inImageRight,
    const cv::Mat& intrinsicParamsLeft,
    const cv::Mat& intrinsicParamsRight,
    const cv::Mat& rightToLeftRotationVector,
    const cv::Mat& rightToLeftTranslationVector,
    const float& minSize = 0.01,
    const float& maxSize = 0.125,
    const double& blockSize = 7,
    const double& offset = 7
    );

} // end namespace

#endif // mitkTagTrackingFacade_h
