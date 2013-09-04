/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkStereoPointProjectionIntoTwoImages.h"
#include "mitkCameraCalibrationFacade.h"
#include <ios>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cv.h>
#include <highgui.h>
#include <niftkFileHelper.h>

namespace mitk {

//-----------------------------------------------------------------------------
StereoPointProjectionIntoTwoImages::StereoPointProjectionIntoTwoImages()
{

}


//-----------------------------------------------------------------------------
StereoPointProjectionIntoTwoImages::~StereoPointProjectionIntoTwoImages()
{

}


//-----------------------------------------------------------------------------
double ComparePointsWithGoldStandard(const std::string& goldStandardFileName, const CvMat* points)
{
  if (goldStandardFileName.size() == 0)
  {
    return 0;
  }

  double rmsError = 0;

  std::ifstream reader(goldStandardFileName.c_str());
  if (!reader)
  {
    std::cerr << "Failed to open " << goldStandardFileName << std::endl;
    return 0;
  }
  else
  {
    std::cout << "Opened " << goldStandardFileName << std::endl;
  }

  std::vector<CvPoint2D32f> pointsIn2D;
  double numbersOnLine[2];

  while(!reader.eof())
  {
    int i = 0;
    for (i = 0; i < 2; i++)
    {
      reader >> numbersOnLine[i];
    }
    if (reader.fail() || i != 2)
    {
      break;
    }
    CvPoint2D32f point;
    point.x = numbersOnLine[0];
    point.y = numbersOnLine[1];
    pointsIn2D.push_back(point);
  }

  if ((int)pointsIn2D.size() != points->rows)
  {
    throw std::logic_error("Gold standard file does not have the right number of points");
  }

  int numberOfPoints = pointsIn2D.size();
  for (int i = 0; i < numberOfPoints; i++)
  {
    CvPoint2D32f goldPoint = pointsIn2D[i];
    CvPoint2D32f testPoint;
    testPoint.x = CV_MAT_ELEM(*points, float, i, 0);
    testPoint.y = CV_MAT_ELEM(*points, float, i, 1);

    double distanceError = (goldPoint.x-testPoint.x)*(goldPoint.x-testPoint.x)
        + (goldPoint.y-testPoint.y)*(goldPoint.y-testPoint.y);
    rmsError += distanceError;

    std::cout << "Point " << i << ", g=(" << goldPoint.x << ", " << goldPoint.y << "), t=(" << testPoint.x << ", " << testPoint.y << "), squaredError=" << distanceError << std::endl;
  }
  rmsError /= (double)(numberOfPoints);
  rmsError = sqrt(rmsError);

  std::cout << "RMS error=" << rmsError << ", compared against " << goldStandardFileName << std::endl;
  return sqrt(rmsError);
}


//-----------------------------------------------------------------------------
bool StereoPointProjectionIntoTwoImages::Project(
    const std::string& input3DFileName,
    const std::string& inputLeftImageName,
    const std::string& inputRightImageName,
    const std::string& outputLeftImageName,
    const std::string& outputRightImageName,
    const std::string& intrinsicLeftFileName,
    const std::string& distortionLeftFileName,
    const std::string& rotationLeftFileName,
    const std::string& translationLeftFileName,
    const std::string& intrinsicRightFileName,
    const std::string& distortionRightFileName,
    const std::string& rightToLeftRotationFileName,
    const std::string& rightToLeftTranslationFileName,
    const std::string& inputLeft2DGoldStandardFileName,
    const std::string& inputRight2DGoldStandardFileName
    )
{
  bool isSuccessful = false;

  try
  {
    IplImage *inputLeftImage = cvLoadImage(inputLeftImageName.c_str());
    if (inputLeftImage == NULL)
    {
      throw std::logic_error("Could not load input left image!");
    }

    IplImage *inputRightImage = cvLoadImage(inputRightImageName.c_str());
    if (inputRightImage == NULL)
    {
      throw std::logic_error("Could not load input right image!");
    }

    CvMat *intrinsicLeft = (CvMat*)cvLoad(intrinsicLeftFileName.c_str());
    if (intrinsicLeft == NULL)
    {
      throw std::logic_error("Failed to load left camera intrinsic params");
    }

    CvMat *distortionLeft = (CvMat*)cvLoad(distortionLeftFileName.c_str());
    if (distortionLeft == NULL)
    {
      throw std::logic_error("Failed to load left camera distortion params");
    }

    CvMat *rotationLeft = (CvMat*)cvLoad(rotationLeftFileName.c_str());
    if (rotationLeft == NULL)
    {
      throw std::logic_error("Failed to load left camera rotation params");
    }

    CvMat *translationLeft = (CvMat*)cvLoad(translationLeftFileName.c_str());
    if (translationLeft == NULL)
    {
      throw std::logic_error("Failed to load left camera translation params");
    }

    CvMat *intrinsicRight = (CvMat*)cvLoad(intrinsicRightFileName.c_str());
    if (intrinsicRight == NULL)
    {
      throw std::logic_error("Failed to load right camera intrinsic params");
    }

    CvMat *distortionRight = (CvMat*)cvLoad(distortionRightFileName.c_str());
    if (distortionRight == NULL)
    {
      throw std::logic_error("Failed to load right camera distortion params");
    }

    CvMat *rightToLeftRotation = (CvMat*)cvLoad(rightToLeftRotationFileName.c_str());
    if (rightToLeftRotation == NULL)
    {
      throw std::logic_error("Failed to load right to left rotation params");
    }

    CvMat *rightToLeftTranslation = (CvMat*)cvLoad(rightToLeftTranslationFileName.c_str());
    if (rightToLeftTranslation == NULL)
    {
      throw std::logic_error("Failed to load right to left translation params");
    }

    // Load the list of 3D points.
    std::ifstream reader(input3DFileName.c_str());
    if (!reader)
    {
      std::cerr << "Failed to open " << input3DFileName << std::endl;
      return false;
    }
    else
    {
      std::cout << "Opened " << input3DFileName << std::endl;
    }

    std::vector<CvPoint3D32f> pointsIn3D;
    double numbersOnLine[3];

    while(!reader.eof())
    {
      int i = 0;
      for (i = 0; i < 3; i++)
      {
        reader >> numbersOnLine[i];
      }
      if (reader.fail() || i != 3)
      {
        break;
      }
      CvPoint3D32f point;
      point.x = numbersOnLine[0];
      point.y = numbersOnLine[1];
      point.z = numbersOnLine[2];
      pointsIn3D.push_back(point);
    }
    reader.close();

    unsigned int numberOfPoints = pointsIn3D.size();

    if (numberOfPoints == 0)
    {
      throw std::logic_error("Failed to read 3D points");
    }
    std::cout << "Read " << numberOfPoints << " points" << std::endl;

    CvMat *points = cvCreateMat(numberOfPoints, 3, CV_32FC1);
    for (unsigned int i = 0; i < numberOfPoints; i++)
    {
      CV_MAT_ELEM(*points, float, i, 0) = pointsIn3D[i].x;
      CV_MAT_ELEM(*points, float, i, 1) = pointsIn3D[i].y;
      CV_MAT_ELEM(*points, float, i, 2) = pointsIn3D[i].z;
    }

    // Save a little memory here.
    pointsIn3D.clear();

    // Allocate output points.
    CvMat *output2DPointsLeft = cvCreateMat(numberOfPoints, 2, CV_32FC1);
    CvMat *output2DPointsRight = cvCreateMat(numberOfPoints, 2, CV_32FC1);

    // Main method to project points to both left and right images.
    Project3DModelPositionsToStereo2D
      (
      *points,
      *intrinsicLeft,
      *distortionLeft,
      *rotationLeft,
      *translationLeft,
      *intrinsicRight,
      *distortionRight,
      *rightToLeftRotation,
      *rightToLeftTranslation,
      *output2DPointsLeft,
      *output2DPointsRight
      );

    for (unsigned int i = 0; i < numberOfPoints; i++)
    {
      std::cout << "[" << i << "], 3D=(" << CV_MAT_ELEM(*points, float, i, 0) \
          << ", " << CV_MAT_ELEM(*points, float, i, 1) \
          << ", " << CV_MAT_ELEM(*points, float, i, 2) \
          << "), 2Dl=(" << CV_MAT_ELEM(*output2DPointsLeft, float, i, 0) \
          << ", " << CV_MAT_ELEM(*output2DPointsLeft, float, i, 1) \
          << "), 2Dr=(" << CV_MAT_ELEM(*output2DPointsRight, float, i, 0) \
          << ", " << CV_MAT_ELEM(*output2DPointsRight, float, i, 1) \
          << ")" << std::endl;
    }

    // Create output image, and draw circles to indicate matched points.

    IplImage *outputLeftImage = cvCloneImage(inputLeftImage);
    IplImage *outputRightImage = cvCloneImage(inputRightImage);

    for (unsigned int i = 0; i < numberOfPoints; i++)
    {
      cvCircle(outputLeftImage, cvPoint(static_cast<int>(CV_MAT_ELEM(*output2DPointsLeft, float, i, 0)), static_cast<int>(CV_MAT_ELEM(*output2DPointsLeft, float, i, 1))), 1, CV_RGB(255,0,0), 1, 8);
      cvCircle(outputLeftImage, cvPoint(static_cast<int>(CV_MAT_ELEM(*output2DPointsLeft, float, i, 0)), static_cast<int>(CV_MAT_ELEM(*output2DPointsLeft, float, i, 1))), 10, CV_RGB(0,255,0), 3, 8);
      cvCircle(outputRightImage, cvPoint(static_cast<int>(CV_MAT_ELEM(*output2DPointsRight, float, i, 0)), static_cast<int>(CV_MAT_ELEM(*output2DPointsRight, float, i, 1))), 1, CV_RGB(255,0,0), 1, 8);
      cvCircle(outputRightImage, cvPoint(static_cast<int>(CV_MAT_ELEM(*output2DPointsRight, float, i, 0)), static_cast<int>(CV_MAT_ELEM(*output2DPointsRight, float, i, 1))), 10, CV_RGB(0,255,0), 3, 8);
    }

    cvSaveImage(outputLeftImageName.c_str(), outputLeftImage);
    cvSaveImage(outputRightImageName.c_str(), outputRightImage);

    // Also, if filenames are specified, compare against lists of points
    ComparePointsWithGoldStandard(inputLeft2DGoldStandardFileName, output2DPointsLeft);
    ComparePointsWithGoldStandard(inputRight2DGoldStandardFileName, output2DPointsRight);

    // Tidy up.
    cvReleaseMat(&points);
    cvReleaseMat(&output2DPointsLeft);
    cvReleaseMat(&output2DPointsRight);
    cvReleaseMat(&intrinsicLeft);
    cvReleaseMat(&distortionLeft);
    cvReleaseMat(&rotationLeft);
    cvReleaseMat(&translationLeft);
    cvReleaseMat(&intrinsicRight);
    cvReleaseMat(&distortionRight);
    cvReleaseMat(&rightToLeftRotation);
    cvReleaseMat(&rightToLeftTranslation);
    cvReleaseImage(&inputLeftImage);
    cvReleaseImage(&inputRightImage);
    cvReleaseImage(&outputLeftImage);
    cvReleaseImage(&outputRightImage);

    isSuccessful = true;
  }
  catch(std::logic_error& e)
  {
    std::cerr << "StereoPointProjectionIntoTwoImages::Project: exception thrown e=" << e.what() << std::endl;
  }

  return isSuccessful;
}

} // end namespace
