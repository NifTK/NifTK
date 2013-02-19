/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkStereoCameraCalibrationFromTwoDirectories.h"
#include "mitkCameraCalibrationFacade.h"
#include <ios>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cv.h>
#include <highgui.h>
#include "FileHelper.h"

namespace mitk {

//-----------------------------------------------------------------------------
StereoCameraCalibrationFromTwoDirectories::StereoCameraCalibrationFromTwoDirectories()
{

}


//-----------------------------------------------------------------------------
StereoCameraCalibrationFromTwoDirectories::~StereoCameraCalibrationFromTwoDirectories()
{

}


//-----------------------------------------------------------------------------
bool StereoCameraCalibrationFromTwoDirectories::Calibrate(const std::string& leftDirectoryName,
    const std::string& rightDirectoryName,
    const int& numberCornersX,
    const int& numberCornersY,
    const float& sizeSquareMillimeters,
    const std::string& outputFileName,
    const bool& writeImages
    )
{
  bool isSuccessful = false;
  int width = 0;
  int height = 0;

  try
  {
    std::vector<IplImage*> imagesLeft;
    std::vector<std::string> fileNamesLeft;
    std::vector<IplImage*> successfullImagesLeft;
    std::vector<std::string> successfullFileNamesLeft;

    std::vector<IplImage*> imagesRight;
    std::vector<std::string> fileNamesRight;
    std::vector<IplImage*> successfullImagesRight;
    std::vector<std::string> successfullFileNamesRight;

    CvMat *imagePointsLeft = NULL;
    CvMat *objectPointsLeft = NULL;
    CvMat *pointCountsLeft = NULL;

    CvMat *imagePointsRight = NULL;
    CvMat *objectPointsRight = NULL;
    CvMat *pointCountsRight = NULL;

    LoadChessBoardsFromDirectory(leftDirectoryName, imagesLeft, fileNamesLeft);
    LoadChessBoardsFromDirectory(rightDirectoryName, imagesRight, fileNamesRight);

    std::vector<IplImage*> allImages;
    allImages.insert(allImages.begin(), imagesLeft.begin(), imagesLeft.end());
    allImages.insert(allImages.begin(), imagesRight.begin(), imagesRight.end());
    CheckConstImageSize(allImages, width, height);

    CvSize imageSize = cvGetSize(allImages[0]);

    ExtractChessBoardPoints(imagesLeft, fileNamesLeft, numberCornersX, numberCornersY, writeImages, successfullImagesLeft, successfullFileNamesLeft, imagePointsLeft, objectPointsLeft, pointCountsLeft);
    ExtractChessBoardPoints(imagesRight, fileNamesRight, numberCornersX, numberCornersY, writeImages, successfullImagesRight, successfullFileNamesRight, imagePointsRight, objectPointsRight, pointCountsRight);

    std::vector<std::string> allSuccessfulFileNames;
    allSuccessfulFileNames.insert(allSuccessfulFileNames.begin(), successfullFileNamesLeft.begin(), successfullFileNamesLeft.end());
    allSuccessfulFileNames.insert(allSuccessfulFileNames.begin(), successfullFileNamesRight.begin(), successfullFileNamesRight.end());

    // Sanity check
    if (successfullImagesLeft.size() != successfullImagesRight.size())
    {
      throw std::logic_error("The left and right channel had a different number of images with successfully matched corners.");
    }

    int numberOfSuccessfulViews = successfullImagesLeft.size();

    CvMat *intrinsicMatrixLeft = cvCreateMat(3,3,CV_32FC1);
    CvMat *distortionCoeffsLeft = cvCreateMat(5, 1, CV_32FC1);
    CvMat *rotationVectorsLeft = cvCreateMat(numberOfSuccessfulViews, 3,CV_32FC1);
    CvMat *translationVectorsLeft = cvCreateMat(numberOfSuccessfulViews, 3, CV_32FC1);

    CvMat *intrinsicMatrixRight = cvCreateMat(3,3,CV_32FC1);
    CvMat *distortionCoeffsRight = cvCreateMat(5, 1, CV_32FC1);
    CvMat *rotationVectorsRight = cvCreateMat(numberOfSuccessfulViews, 3,CV_32FC1);
    CvMat *translationVectorsRight = cvCreateMat(numberOfSuccessfulViews, 3, CV_32FC1);

    CvMat *rightToLeftRotationMatrix = cvCreateMat(3, 3,CV_32FC1);
    CvMat *rightToLeftTranslationVector = cvCreateMat(3, 1, CV_32FC1);
    CvMat *essentialMatrix = cvCreateMat(3, 3,CV_32FC1);
    CvMat *fundamentalMatrix = cvCreateMat(3, 3,CV_32FC1);

    double projectionError = CalibrateStereoCameraParameters(
        numberOfSuccessfulViews,
        *objectPointsLeft,
        *imagePointsLeft,
        *pointCountsLeft,
        imageSize,
        *objectPointsRight,
        *imagePointsRight,
        *pointCountsRight,
        *intrinsicMatrixLeft,
        *distortionCoeffsLeft,
        *rotationVectorsLeft,
        *translationVectorsLeft,
        *intrinsicMatrixRight,
        *distortionCoeffsRight,
        *rotationVectorsRight,
        *translationVectorsRight,
        *rightToLeftRotationMatrix,
        *rightToLeftTranslationVector,
        *essentialMatrix,
        *fundamentalMatrix
        );

    std::ostream *os = NULL;
    std::ostringstream oss;
    std::ofstream fs;

    if (outputFileName.size() > 0)
    {
      fs.open(outputFileName.c_str(), std::ios::out);
      if (!fs.fail())
      {
        os = &fs;
        std::cout << "Writing to " << outputFileName << std::endl;
      }
      else
      {
        std::cerr << "ERROR: Writing calibration data to file " << outputFileName << " failed!" << std::endl;
      }
    }
    else
    {
      os = &oss;
    }

    *os << "Stereo calibration" << std::endl;
    float zero = 0.0f;
    float one = 1.0;

    *os << CV_MAT_ELEM(*rightToLeftRotationMatrix, float, 0, 0) << ", " << CV_MAT_ELEM(*rightToLeftRotationMatrix, float, 0, 1) << ", " << CV_MAT_ELEM(*rightToLeftRotationMatrix, float, 0, 2) << ", " << CV_MAT_ELEM(*rightToLeftTranslationVector, float, 0, 0) << std::endl;
    *os << CV_MAT_ELEM(*rightToLeftRotationMatrix, float, 1, 0) << ", " << CV_MAT_ELEM(*rightToLeftRotationMatrix, float, 1, 1) << ", " << CV_MAT_ELEM(*rightToLeftRotationMatrix, float, 1, 2) << ", " << CV_MAT_ELEM(*rightToLeftTranslationVector, float, 1, 0) << std::endl;
    *os << CV_MAT_ELEM(*rightToLeftRotationMatrix, float, 2, 0) << ", " << CV_MAT_ELEM(*rightToLeftRotationMatrix, float, 2, 1) << ", " << CV_MAT_ELEM(*rightToLeftRotationMatrix, float, 2, 2) << ", " << CV_MAT_ELEM(*rightToLeftTranslationVector, float, 2, 0) << std::endl;
    *os << zero << ", " << zero << ", " << zero << ", " << one << std::endl;

    *os << "Essential matrix" << std::endl;
    *os << CV_MAT_ELEM(*essentialMatrix, float, 0, 0) << ", " << CV_MAT_ELEM(*essentialMatrix, float, 0, 1) << ", " << CV_MAT_ELEM(*essentialMatrix, float, 0, 2) << std::endl;
    *os << CV_MAT_ELEM(*essentialMatrix, float, 1, 0) << ", " << CV_MAT_ELEM(*essentialMatrix, float, 1, 1) << ", " << CV_MAT_ELEM(*essentialMatrix, float, 1, 2) << std::endl;
    *os << CV_MAT_ELEM(*essentialMatrix, float, 2, 0) << ", " << CV_MAT_ELEM(*essentialMatrix, float, 2, 1) << ", " << CV_MAT_ELEM(*essentialMatrix, float, 2, 2) << std::endl;

    *os << "Fundamental matrix" << std::endl;
    *os << CV_MAT_ELEM(*fundamentalMatrix, float, 0, 0) << ", " << CV_MAT_ELEM(*fundamentalMatrix, float, 0, 1) << ", " << CV_MAT_ELEM(*fundamentalMatrix, float, 0, 2) << std::endl;
    *os << CV_MAT_ELEM(*fundamentalMatrix, float, 1, 0) << ", " << CV_MAT_ELEM(*fundamentalMatrix, float, 1, 1) << ", " << CV_MAT_ELEM(*fundamentalMatrix, float, 1, 2) << std::endl;
    *os << CV_MAT_ELEM(*fundamentalMatrix, float, 2, 0) << ", " << CV_MAT_ELEM(*fundamentalMatrix, float, 2, 1) << ", " << CV_MAT_ELEM(*fundamentalMatrix, float, 2, 2) << std::endl;

    *os << "Left camera" << std::endl;
    OutputCalibrationData(
        *os,
        *objectPointsLeft,
        *imagePointsLeft,
        *pointCountsLeft,
        *intrinsicMatrixLeft,
        *distortionCoeffsLeft,
        *rotationVectorsLeft,
        *translationVectorsLeft,
        projectionError,
        width,
        height,
        numberCornersX,
        numberCornersY,
        successfullFileNamesLeft
        );

    // Also output these as XML, as they are used in niftkCorrectVideoDistortion
    cvSave(std::string(outputFileName + ".left.intrinsic.xml").c_str(), intrinsicMatrixLeft);
    cvSave(std::string(outputFileName + ".left.distortion.xml").c_str(), distortionCoeffsLeft);

    *os << "Right camera" << std::endl;
    OutputCalibrationData(
        *os,
        *objectPointsRight,
        *imagePointsRight,
        *pointCountsRight,
        *intrinsicMatrixRight,
        *distortionCoeffsRight,
        *rotationVectorsRight,
        *translationVectorsRight,
        projectionError,
        width,
        height,
        numberCornersX,
        numberCornersY,
        successfullFileNamesRight
        );

    // Also output these as XML, as they are used in niftkCorrectVideoDistortion
    cvSave(std::string(outputFileName + ".right.intrinsic.xml").c_str(), intrinsicMatrixRight);
    cvSave(std::string(outputFileName + ".right.distortion.xml").c_str(), distortionCoeffsRight);

    // Tidy up.
    if(fs.is_open())
    {
      fs.close();
    }

    cvReleaseMat(&imagePointsLeft);
    cvReleaseMat(&objectPointsLeft);
    cvReleaseMat(&pointCountsLeft);
    cvReleaseMat(&imagePointsRight);
    cvReleaseMat(&objectPointsRight);
    cvReleaseMat(&pointCountsRight);

    cvReleaseMat(&intrinsicMatrixLeft);
    cvReleaseMat(&distortionCoeffsLeft);
    cvReleaseMat(&rotationVectorsLeft);
    cvReleaseMat(&translationVectorsLeft);

    cvReleaseMat(&intrinsicMatrixRight);
    cvReleaseMat(&distortionCoeffsRight);
    cvReleaseMat(&rotationVectorsRight);
    cvReleaseMat(&translationVectorsRight);

    cvReleaseMat(&rightToLeftRotationMatrix);
    cvReleaseMat(&rightToLeftTranslationVector);
    cvReleaseMat(&fundamentalMatrix);
    cvReleaseMat(&essentialMatrix);

    for (unsigned int i = 0; i < allImages.size(); i++)
    {
      cvReleaseImage(&allImages[i]);
    }

    isSuccessful = true;
  }
  catch(std::logic_error e)
  {
    std::cerr << "CameraCalibrationFromDirectory::Calibrate: exception thrown e=" << e.what() << std::endl;
  }

  return isSuccessful;
}

} // end namespace
