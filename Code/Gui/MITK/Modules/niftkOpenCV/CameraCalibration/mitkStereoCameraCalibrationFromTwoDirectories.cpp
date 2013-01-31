/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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

    CvMat *intrinsicMatrixLeft = cvCreateMat(3,3,CV_32FC1);
    CvMat *distortionCoeffsLeft = cvCreateMat(5, 1, CV_32FC1);
    CvMat *rotationMatrixLeft = cvCreateMat(3, 3,CV_32FC1);
    CvMat *translationVectorLeft = cvCreateMat(3, 1, CV_32FC1);

    CvMat *intrinsicMatrixRight = cvCreateMat(3,3,CV_32FC1);
    CvMat *distortionCoeffsRight = cvCreateMat(5, 1, CV_32FC1);
    CvMat *rotationMatrixRight = cvCreateMat(3, 3,CV_32FC1);
    CvMat *translationVectorRight = cvCreateMat(3, 1, CV_32FC1);

    CvMat *leftToRightRotationMatrix = cvCreateMat(3, 3,CV_32FC1);
    CvMat *leftToRightTranslationVector = cvCreateMat(3, 1, CV_32FC1);

    double projectionError = CalibrateStereoCameraParameters(
        successfullImagesLeft.size(),
        *objectPointsLeft,
        *imagePointsLeft,
        *pointCountsLeft,
        imageSize,
        *objectPointsRight,
        *imagePointsRight,
        *pointCountsRight,
        *intrinsicMatrixLeft,
        *distortionCoeffsLeft,
        *rotationMatrixLeft,
        *translationVectorLeft,
        *intrinsicMatrixRight,
        *distortionCoeffsRight,
        *rotationMatrixRight,
        *translationVectorRight,
        *leftToRightRotationMatrix,
        *leftToRightTranslationVector
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

    // Output Calibration Data.

    float zero = 0;
    float one = 1;

    *os << CV_MAT_ELEM(*intrinsicMatrixLeft, float, 0, 0) << "," << CV_MAT_ELEM(*intrinsicMatrixLeft, float, 0, 1) << "," << CV_MAT_ELEM(*intrinsicMatrixLeft, float, 0, 2) << std::endl;
    *os << CV_MAT_ELEM(*intrinsicMatrixLeft, float, 1, 0) << "," << CV_MAT_ELEM(*intrinsicMatrixLeft, float, 1, 1) << "," << CV_MAT_ELEM(*intrinsicMatrixLeft, float, 1, 2) << std::endl;
    *os << CV_MAT_ELEM(*intrinsicMatrixLeft, float, 2, 0) << "," << CV_MAT_ELEM(*intrinsicMatrixLeft, float, 2, 1) << "," << CV_MAT_ELEM(*intrinsicMatrixLeft, float, 2, 2) << std::endl;
    *os << CV_MAT_ELEM(*rotationMatrixLeft, float, 0, 0) << "," << CV_MAT_ELEM(*rotationMatrixLeft, float, 0, 1) << "," << CV_MAT_ELEM(*rotationMatrixLeft, float, 0, 2) << "," << CV_MAT_ELEM(*translationVectorLeft, float, 0, 0) << std::endl;
    *os << CV_MAT_ELEM(*rotationMatrixLeft, float, 1, 0) << "," << CV_MAT_ELEM(*rotationMatrixLeft, float, 1, 1) << "," << CV_MAT_ELEM(*rotationMatrixLeft, float, 1, 2) << "," << CV_MAT_ELEM(*translationVectorLeft, float, 1, 0) << std::endl;
    *os << CV_MAT_ELEM(*rotationMatrixLeft, float, 2, 0) << "," << CV_MAT_ELEM(*rotationMatrixLeft, float, 2, 1) << "," << CV_MAT_ELEM(*rotationMatrixLeft, float, 2, 2) << "," << CV_MAT_ELEM(*translationVectorLeft, float, 2, 0) << std::endl;
    *os << zero << "," << zero << "," << zero << "," << one << std::endl;
    *os << CV_MAT_ELEM(*distortionCoeffsLeft, float, 0, 0) << "," << CV_MAT_ELEM(*distortionCoeffsLeft, float, 1, 0) << "," << CV_MAT_ELEM(*distortionCoeffsLeft, float, 2, 0) << "," << CV_MAT_ELEM(*distortionCoeffsLeft, float, 3, 0) << "," << CV_MAT_ELEM(*distortionCoeffsLeft, float, 4, 0) << std::endl;

    *os << CV_MAT_ELEM(*intrinsicMatrixRight, float, 0, 0) << "," << CV_MAT_ELEM(*intrinsicMatrixRight, float, 0, 1) << "," << CV_MAT_ELEM(*intrinsicMatrixRight, float, 0, 2) << std::endl;
    *os << CV_MAT_ELEM(*intrinsicMatrixRight, float, 1, 0) << "," << CV_MAT_ELEM(*intrinsicMatrixRight, float, 1, 1) << "," << CV_MAT_ELEM(*intrinsicMatrixRight, float, 1, 2) << std::endl;
    *os << CV_MAT_ELEM(*intrinsicMatrixRight, float, 2, 0) << "," << CV_MAT_ELEM(*intrinsicMatrixRight, float, 2, 1) << "," << CV_MAT_ELEM(*intrinsicMatrixRight, float, 2, 2) << std::endl;
    *os << CV_MAT_ELEM(*rotationMatrixRight, float, 0, 0) << "," << CV_MAT_ELEM(*rotationMatrixRight, float, 0, 1) << "," << CV_MAT_ELEM(*rotationMatrixRight, float, 0, 2) << "," << CV_MAT_ELEM(*translationVectorRight, float, 0, 0) << std::endl;
    *os << CV_MAT_ELEM(*rotationMatrixRight, float, 1, 0) << "," << CV_MAT_ELEM(*rotationMatrixRight, float, 1, 1) << "," << CV_MAT_ELEM(*rotationMatrixRight, float, 1, 2) << "," << CV_MAT_ELEM(*translationVectorRight, float, 1, 0) << std::endl;
    *os << CV_MAT_ELEM(*rotationMatrixRight, float, 2, 0) << "," << CV_MAT_ELEM(*rotationMatrixRight, float, 2, 1) << "," << CV_MAT_ELEM(*rotationMatrixRight, float, 2, 2) << "," << CV_MAT_ELEM(*translationVectorRight, float, 2, 0) << std::endl;
    *os << zero << "," << zero << "," << zero << "," << one << std::endl;
    *os << CV_MAT_ELEM(*distortionCoeffsRight, float, 0, 0) << "," << CV_MAT_ELEM(*distortionCoeffsRight, float, 1, 0) << "," << CV_MAT_ELEM(*distortionCoeffsRight, float, 2, 0) << "," << CV_MAT_ELEM(*distortionCoeffsRight, float, 3, 0) << "," << CV_MAT_ELEM(*distortionCoeffsRight, float, 4, 0) << std::endl;

    *os << CV_MAT_ELEM(*leftToRightRotationMatrix, float, 0, 0) << "," << CV_MAT_ELEM(*leftToRightRotationMatrix, float, 0, 1) << "," << CV_MAT_ELEM(*leftToRightRotationMatrix, float, 0, 2) << "," << CV_MAT_ELEM(*leftToRightRotationMatrix, float, 0, 0) << std::endl;
    *os << CV_MAT_ELEM(*leftToRightRotationMatrix, float, 1, 0) << "," << CV_MAT_ELEM(*leftToRightRotationMatrix, float, 1, 1) << "," << CV_MAT_ELEM(*leftToRightRotationMatrix, float, 1, 2) << "," << CV_MAT_ELEM(*leftToRightRotationMatrix, float, 1, 0) << std::endl;
    *os << CV_MAT_ELEM(*leftToRightRotationMatrix, float, 2, 0) << "," << CV_MAT_ELEM(*leftToRightRotationMatrix, float, 2, 1) << "," << CV_MAT_ELEM(*leftToRightRotationMatrix, float, 2, 2) << "," << CV_MAT_ELEM(*leftToRightRotationMatrix, float, 2, 0) << std::endl;
    *os << zero << "," << zero << "," << zero << "," << one << std::endl;

    *os << "projection error:" << projectionError << std::endl;
    *os << "image size:" << width << " " << height << std::endl;
    *os << "number of internal corners:" << numberCornersX << " " << numberCornersY << std::endl;

    *os << "number of files used:" << allSuccessfulFileNames.size() << std::endl;
    *os << "list of files used:" << std::endl;

    // Also output files actually used.
    for (unsigned int i = 0; i < allSuccessfulFileNames.size(); i++)
    {
      *os << allSuccessfulFileNames[i] << std::endl;
    }

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
    cvReleaseMat(&rotationMatrixLeft);
    cvReleaseMat(&translationVectorLeft);

    cvReleaseMat(&intrinsicMatrixRight);
    cvReleaseMat(&distortionCoeffsRight);
    cvReleaseMat(&rotationMatrixRight);
    cvReleaseMat(&translationVectorRight);

    cvReleaseMat(&leftToRightRotationMatrix);
    cvReleaseMat(&leftToRightTranslationVector);

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
