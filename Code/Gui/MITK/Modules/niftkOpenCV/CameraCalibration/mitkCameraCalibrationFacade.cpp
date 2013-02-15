/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkCameraCalibrationFacade.h"
#include "mitkStereoDistortionCorrectionVideoProcessor.h"
#include "FileHelper.h"
#include <iostream>
#include <cv.h>
#include <highgui.h>

namespace mitk {

//-----------------------------------------------------------------------------
void LoadChessBoardsFromDirectory(const std::string& fullDirectoryName,
                                                    std::vector<IplImage*>& images,
                                                    std::vector<std::string>& fileNames)
{
  std::vector<std::string> files = niftk::GetFilesInDirectory(fullDirectoryName);
  if (files.size() > 0)
  {
    for(unsigned int i = 0; i < files.size();i++)
    {
      IplImage* image = cvLoadImage(files[i].c_str());
      if (image != NULL)
      {
        images.push_back(image);
        fileNames.push_back(files[i]);
      }
    }
  }
  else
  {
    throw std::logic_error("No files found in directory!");
  }

  if (images.size() == 0)
  {
    throw std::logic_error("No images found in directory!");
  }

  std::cout << "Loaded " << fileNames.size() << " chess boards from " << fullDirectoryName << std::endl;
}


//-----------------------------------------------------------------------------
void CheckConstImageSize(const std::vector<IplImage*>& images, int& width, int& height)
{
  width = 0;
  height = 0;

  if (images.size() == 0)
  {
    throw std::logic_error("Vector of images is empty!");
  }

  width = images[0]->width;
  height = images[0]->height;

  for (unsigned int i = 1; i < images.size(); i++)
  {
    if (images[i]->width != width || images[i]->height != height)
    {
      throw std::logic_error("Images are of inconsistent sizes!");
    }
  }

  std::cout << "Chess board images are (" << width << ", " << height << ") pixels" << std::endl;
}


//-----------------------------------------------------------------------------
void ExtractChessBoardPoints(const std::vector<IplImage*>& images,
                             const std::vector<std::string>& fileNames,
                             const int& numberCornersWidth,
                             const int& numberCornersHeight,
                             const bool& drawCorners,
                             std::vector<IplImage*>& outputImages,
                             std::vector<std::string>& outputFileNames,
                             CvMat*& outputImagePoints,
                             CvMat*& outputObjectPoints,
                             CvMat*& outputPointCounts
                             )
{

  if (images.size() != fileNames.size())
  {
    throw std::logic_error("The list of images and list of filenames have different lengths!");
  }

  outputImages.clear();
  outputFileNames.clear();

  unsigned int numberOfChessBoards = images.size();
  unsigned int numberOfCorners = numberCornersWidth * numberCornersHeight;
  CvSize boardSize = cvSize(numberCornersWidth, numberCornersHeight);

  std::cout << "Searching for " << numberCornersWidth << " x " << numberCornersHeight << " = " << numberOfCorners << std::endl;

  CvMat* imagePoints  = cvCreateMat(numberOfChessBoards * numberOfCorners, 2, CV_32FC1);
  CvMat* objectPoints = cvCreateMat(numberOfChessBoards * numberOfCorners, 3, CV_32FC1);
  CvMat* pointCounts = cvCreateMat(numberOfChessBoards, 1, CV_32FC1);
  CvPoint2D32f* corners = new CvPoint2D32f[numberOfCorners];

  int cornerCount = 0;
  int successes = 0;
  int step = 0;

  IplImage *greyImage = cvCreateImage(cvGetSize(images[0]), 8, 1);

  // Iterate over each image, finding corners.
  for (unsigned int i = 0; i < images.size(); i++)
  {
    std::cout << "Processing file " << fileNames[i] << std::endl;

    int found = cvFindChessboardCorners(images[i], boardSize, corners, &cornerCount,
        CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

    // Get sub-pixel accuracy.
    cvCvtColor(images[i], greyImage, CV_BGR2GRAY);
    cvFindCornerSubPix(greyImage, corners, cornerCount, cvSize(11,11), cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));

    if (drawCorners)
    {
      std::string outputName = fileNames[i] + ".output.jpg";
      cvDrawChessboardCorners(images[i], boardSize, corners, cornerCount, found);
      cvSaveImage(outputName.c_str(), images[i]);
    }

    // If we got the right number of corners, add it to our data.
    if (found != 0 && cornerCount == (int)numberOfCorners)
    {
      step = successes * numberOfCorners;
      for (int j=step, k=0; k<(int)numberOfCorners; ++j, ++k)
      {
        CV_MAT_ELEM(*imagePoints, float, j, 0) = corners[k].x;
        CV_MAT_ELEM(*imagePoints, float, j, 1) = corners[k].y;
        CV_MAT_ELEM(*objectPoints, float, j, 0) = k/numberCornersWidth;
        CV_MAT_ELEM(*objectPoints, float, j, 1) = k%numberCornersWidth;
        CV_MAT_ELEM(*objectPoints, float, j, 2) = 0;
      }
      CV_MAT_ELEM(*pointCounts, int, successes, 0) = numberOfCorners;
      successes++;

      outputImages.push_back(images[i]);
      outputFileNames.push_back(fileNames[i]);
    }

    std::cout << "Processed image " << i << ", corners=" << cornerCount << ", found=" << found << ", successes=" << successes << std::endl;
  }

  if (successes == 0)
  {
    throw std::logic_error("The chessboard feature detection failed");
  }

  // Now re-allocate points based on what we found.
  outputObjectPoints = cvCreateMat(successes*numberOfCorners,3,CV_32FC1);
  outputImagePoints  = cvCreateMat(successes*numberOfCorners,2,CV_32FC1);
  outputPointCounts  = cvCreateMat(successes,1,CV_32SC1);

  for (int i = 0; i < successes*(int)numberOfCorners; ++i)
  {
    CV_MAT_ELEM(*outputImagePoints, float, i, 0) = CV_MAT_ELEM(*imagePoints, float, i, 0);
    CV_MAT_ELEM(*outputImagePoints, float, i, 1) = CV_MAT_ELEM(*imagePoints, float, i, 1);
    CV_MAT_ELEM(*outputObjectPoints, float, i, 0) = CV_MAT_ELEM(*objectPoints, float, i, 0);
    CV_MAT_ELEM(*outputObjectPoints, float, i, 1) = CV_MAT_ELEM(*objectPoints, float, i, 1);
    CV_MAT_ELEM(*outputObjectPoints, float, i, 2) = CV_MAT_ELEM(*objectPoints, float, i, 2);
  }
  for (int i = 0; i < successes; ++i)
  {
    CV_MAT_ELEM(*outputPointCounts, int, i, 0) = CV_MAT_ELEM(*pointCounts, int, i, 0);
  }

  cvReleaseMat(&objectPoints);
  cvReleaseMat(&imagePoints);
  cvReleaseMat(&pointCounts);
  cvReleaseImage(&greyImage);
  delete [] corners;

  std::cout << "Successfully processed " << successes << " out of " << images.size() << std::endl;
}


//-----------------------------------------------------------------------------
double CalibrateSingleCameraIntrinsicParameters(
    const CvMat&  objectPoints,
    const CvMat&  imagePoints,
    const CvMat&  pointCounts,
    const CvSize& imageSize,
    CvMat&  outputIntrinsicMatrix,
    CvMat&  outputDistortionCoefficients,
    const int& flags
    )
{
  return cvCalibrateCamera2(&objectPoints,
                            &imagePoints,
                            &pointCounts,
                            imageSize,
                            &outputIntrinsicMatrix,
                            &outputDistortionCoefficients,
                            NULL,
                            NULL,
                            flags
                            );

}


//-----------------------------------------------------------------------------
double CalibrateSingleCameraIntrinsicUsing3Passes(
       const CvMat& objectPoints,
       const CvMat& imagePoints,
       const CvMat& pointCounts,
       const CvSize& imageSize,
       CvMat& outputIntrinsicMatrix,
       CvMat& outputDistortionCoefficients
       )
{
  CvScalar zero = cvScalar(0);
  cvSet(&outputIntrinsicMatrix, zero);
  cvSet(&outputDistortionCoefficients, zero);

  CV_MAT_ELEM(outputIntrinsicMatrix, float, 0, 0) = 1.0f;
  CV_MAT_ELEM(outputIntrinsicMatrix, float, 1, 1) = 1.0f;

  double reprojectionError1 = CalibrateSingleCameraIntrinsicParameters(
      objectPoints, imagePoints, pointCounts, imageSize, outputIntrinsicMatrix, outputDistortionCoefficients,
      CV_CALIB_FIX_PRINCIPAL_POINT | CV_CALIB_FIX_ASPECT_RATIO
      );

  double reprojectionError2 = CalibrateSingleCameraIntrinsicParameters(
      objectPoints, imagePoints, pointCounts, imageSize, outputIntrinsicMatrix, outputDistortionCoefficients,
      CV_CALIB_FIX_PRINCIPAL_POINT
      );

  double reprojectionError3 = CalibrateSingleCameraIntrinsicParameters(
      objectPoints, imagePoints, pointCounts, imageSize, outputIntrinsicMatrix, outputDistortionCoefficients
      );

  std::cout << "3 pass intrinsic calibration yielded RPE of " << reprojectionError1 << ", " << reprojectionError2 << ", " << reprojectionError3 << std::endl;

  return reprojectionError3;
}


//-----------------------------------------------------------------------------
void CalibrateSingleCameraExtrinsicParameters(
    const CvMat& objectPoints,
    const CvMat& imagePoints,
    const CvMat& intrinsicMatrix,
    const CvMat& distortionCoefficients,
    CvMat& outputRotationMatrix,
    CvMat& outputTranslationVector
    )
{
  CvMat *rotationVector = cvCreateMat(3, 1, CV_32FC1);

  cvFindExtrinsicCameraParams2(&objectPoints,
      &imagePoints,
      &intrinsicMatrix,
      &distortionCoefficients,
      rotationVector,
      &outputTranslationVector
      );

  cvRodrigues2(rotationVector, &outputRotationMatrix);
  cvReleaseMat(&rotationVector);
}


//-----------------------------------------------------------------------------
double CalibrateSingleCameraParameters(
     const int& numberSuccessfulViews,
     const CvMat& objectPoints,
     const CvMat& imagePoints,
     const CvMat& pointCounts,
     const CvSize& imageSize,
     CvMat& outputIntrinsicMatrix,
     CvMat& outputDistortionCoefficients,
     CvMat& outputRotationVectors,
     CvMat& outputTranslationVectors
     )
{

  double reprojectionError1 = CalibrateSingleCameraIntrinsicUsing3Passes
      (
       objectPoints,
       imagePoints,
       pointCounts,
       imageSize,
       outputIntrinsicMatrix,
       outputDistortionCoefficients
      );

  double reprojectionError2 = cvCalibrateCamera2(
                            &objectPoints,
                            &imagePoints,
                            &pointCounts,
                            imageSize,
                            &outputIntrinsicMatrix,
                            &outputDistortionCoefficients,
                            &outputRotationVectors,
                            &outputTranslationVectors,
                            CV_CALIB_USE_INTRINSIC_GUESS // This assumes you have called
                                                         // CalibrateSingleCameraIntrinsicUsing3Passes
                            );
  std::cout << "3 pass intrinsic, then intrinsic+extrinsic yielded RPE of " << reprojectionError1 << ", " << reprojectionError2 << std::endl;

  return reprojectionError2;
}


//-----------------------------------------------------------------------------
void ExtractExtrinsicMatrixFromRotationAndTranslationVectors(
    const CvMat& rotationVectors,
    const CvMat& translationVectors,
    const int& viewNumber,
    CvMat& outputExtrinsicMatrix
    )
{
  CvMat *rotationVector = cvCreateMat(1, 3, CV_32FC1);
  CvMat *translationVector = cvCreateMat(1, 3, CV_32FC1);
  CvMat *rotationMatrix = cvCreateMat(3, 3, CV_32FC1);

  CV_MAT_ELEM(*rotationVector, float, 0, 0) = CV_MAT_ELEM(rotationVectors, float, viewNumber, 0);
  CV_MAT_ELEM(*rotationVector, float, 0, 1) = CV_MAT_ELEM(rotationVectors, float, viewNumber, 1);
  CV_MAT_ELEM(*rotationVector, float, 0, 2) = CV_MAT_ELEM(rotationVectors, float, viewNumber, 2);

  CV_MAT_ELEM(*translationVector, float, 0, 0) = CV_MAT_ELEM(translationVectors, float, viewNumber, 0);
  CV_MAT_ELEM(*translationVector, float, 0, 1) = CV_MAT_ELEM(translationVectors, float, viewNumber, 1);
  CV_MAT_ELEM(*translationVector, float, 0, 2) = CV_MAT_ELEM(translationVectors, float, viewNumber, 2);

  cvRodrigues2(rotationVector, rotationMatrix);

  CV_MAT_ELEM(outputExtrinsicMatrix, float, 0, 0) = CV_MAT_ELEM(*rotationMatrix, float, 0, 0);
  CV_MAT_ELEM(outputExtrinsicMatrix, float, 0, 1) = CV_MAT_ELEM(*rotationMatrix, float, 0, 1);
  CV_MAT_ELEM(outputExtrinsicMatrix, float, 0, 2) = CV_MAT_ELEM(*rotationMatrix, float, 0, 2);
  CV_MAT_ELEM(outputExtrinsicMatrix, float, 1, 0) = CV_MAT_ELEM(*rotationMatrix, float, 1, 0);
  CV_MAT_ELEM(outputExtrinsicMatrix, float, 1, 1) = CV_MAT_ELEM(*rotationMatrix, float, 1, 1);
  CV_MAT_ELEM(outputExtrinsicMatrix, float, 1, 2) = CV_MAT_ELEM(*rotationMatrix, float, 1, 2);
  CV_MAT_ELEM(outputExtrinsicMatrix, float, 2, 0) = CV_MAT_ELEM(*rotationMatrix, float, 2, 0);
  CV_MAT_ELEM(outputExtrinsicMatrix, float, 2, 1) = CV_MAT_ELEM(*rotationMatrix, float, 2, 1);
  CV_MAT_ELEM(outputExtrinsicMatrix, float, 2, 2) = CV_MAT_ELEM(*rotationMatrix, float, 2, 2);
  CV_MAT_ELEM(outputExtrinsicMatrix, float, 0, 3) = CV_MAT_ELEM(*translationVector, float, 0, 0);
  CV_MAT_ELEM(outputExtrinsicMatrix, float, 1, 3) = CV_MAT_ELEM(*translationVector, float, 0, 1);
  CV_MAT_ELEM(outputExtrinsicMatrix, float, 2, 3) = CV_MAT_ELEM(*translationVector, float, 0, 2);
  CV_MAT_ELEM(outputExtrinsicMatrix, float, 3, 0) = 0.0f;
  CV_MAT_ELEM(outputExtrinsicMatrix, float, 3, 1) = 0.0f;
  CV_MAT_ELEM(outputExtrinsicMatrix, float, 3, 2) = 0.0f;
  CV_MAT_ELEM(outputExtrinsicMatrix, float, 3, 3) = 1.0f;

  cvReleaseMat(&rotationVector);
  cvReleaseMat(&translationVector);
  cvReleaseMat(&rotationMatrix);
}


//-----------------------------------------------------------------------------
void ProjectAllPoints(
    const int& numberSuccessfulViews,
    const int& pointCount,
    const CvMat& objectPoints,
    const CvMat& imagePoints,
    const CvMat& intrinsicMatrix,
    const CvMat& distortionCoeffictions,
    const CvMat& rotationVectors,
    const CvMat& translationVectors,
    CvMat& outputImagePoints
    )
{
  CvMat *rotationVector = cvCreateMat(1, 3, CV_32FC1);
  CvMat *translationVector = cvCreateMat(1, 3, CV_32FC1);
  CvMat *objectPointsFor1View = cvCreateMat(pointCount, 3, CV_32FC1);
  CvMat *imagePointsFor1View = cvCreateMat(pointCount, 2, CV_32FC1);

  for (int i = 0; i < numberSuccessfulViews; i++)
  {
    for (int j = 0; j < pointCount; j++)
    {
      CV_MAT_ELEM(*objectPointsFor1View, float, j, 0) = CV_MAT_ELEM(objectPoints, float, i*pointCount + j, 0);
      CV_MAT_ELEM(*objectPointsFor1View, float, j, 1) = CV_MAT_ELEM(objectPoints, float, i*pointCount + j, 1);
      CV_MAT_ELEM(*objectPointsFor1View, float, j, 2) = CV_MAT_ELEM(objectPoints, float, i*pointCount + j, 2);
    }
    CV_MAT_ELEM(*rotationVector, float, 0, 0) = CV_MAT_ELEM(rotationVectors, float, i, 0);
    CV_MAT_ELEM(*rotationVector, float, 0, 1) = CV_MAT_ELEM(rotationVectors, float, i, 1);
    CV_MAT_ELEM(*rotationVector, float, 0, 2) = CV_MAT_ELEM(rotationVectors, float, i, 2);
    CV_MAT_ELEM(*translationVector, float, 0, 0) = CV_MAT_ELEM(translationVectors, float, i, 0);
    CV_MAT_ELEM(*translationVector, float, 0, 1) = CV_MAT_ELEM(translationVectors, float, i, 1);
    CV_MAT_ELEM(*translationVector, float, 0, 2) = CV_MAT_ELEM(translationVectors, float, i, 2);

    cvProjectPoints2(
        objectPointsFor1View,
        rotationVector,
        translationVector,
        &intrinsicMatrix,
        &distortionCoeffictions,
        imagePointsFor1View
        );

    for (int j = 0; j < pointCount; j++)
    {
      CV_MAT_ELEM(outputImagePoints, float, i*pointCount + j, 0) = CV_MAT_ELEM(*imagePointsFor1View, float, j, 0);
      CV_MAT_ELEM(outputImagePoints, float, i*pointCount + j, 1) = CV_MAT_ELEM(*imagePointsFor1View, float, j, 1);
    }
  }

  cvReleaseMat(&rotationVector);
  cvReleaseMat(&translationVector);
  cvReleaseMat(&objectPointsFor1View);
  cvReleaseMat(&imagePointsFor1View);
}


//-----------------------------------------------------------------------------
double CalibrateStereoCameraParameters(
    const int& numberSuccessfulViews,
    const CvMat& objectPointsLeft,
    const CvMat& imagePointsLeft,
    const CvMat& pointCountsLeft,
    const CvSize& imageSize,
    const CvMat& objectPointsRight,
    const CvMat& imagePointsRight,
    const CvMat& pointCountsRight,
    CvMat& outputIntrinsicMatrixLeft,
    CvMat& outputDistortionCoefficientsLeft,
    CvMat& outputRotationVectorsLeft,
    CvMat& outputTranslationVectorsLeft,
    CvMat& outputIntrinsicMatrixRight,
    CvMat& outputDistortionCoefficientsRight,
    CvMat& outputRotationVectorsRight,
    CvMat& outputTranslationVectorsRight,
    CvMat& outputRightToLeftRotation,
    CvMat& outputRightToLeftTranslation,
    CvMat& outputEssentialMatrix,
    CvMat& outputFundamentalMatrix
    )
{
  double leftProjectionError = CalibrateSingleCameraIntrinsicUsing3Passes(
      objectPointsLeft,
      imagePointsLeft,
      pointCountsLeft,
      imageSize,
      outputIntrinsicMatrixLeft,
      outputDistortionCoefficientsLeft
      );

  double rightProjectionError = CalibrateSingleCameraIntrinsicUsing3Passes(
      objectPointsRight,
      imagePointsRight,
      pointCountsRight,
      imageSize,
      outputIntrinsicMatrixRight,
      outputDistortionCoefficientsRight);

  std::cout << "Initial intrinsic calibration gave re-projection errors of left=" << leftProjectionError << ", right=" << rightProjectionError << std::endl;

  double stereoCalibrationProjectionError = cvStereoCalibrate
      (
      &objectPointsLeft,
      &imagePointsLeft,
      &imagePointsRight,
      &pointCountsLeft,
      &outputIntrinsicMatrixLeft,
      &outputDistortionCoefficientsLeft,
      &outputIntrinsicMatrixRight,
      &outputDistortionCoefficientsRight,
      imageSize,
      &outputRightToLeftRotation,
      &outputRightToLeftTranslation,
      &outputEssentialMatrix,
      &outputFundamentalMatrix,
      cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, 1e-6), // where cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, 1e-6) is the default.
      CV_CALIB_FIX_INTRINSIC // Use the initial guess, but feel free to change optimise it.
      );

  std::cout << "Stereo re-projection error=" << stereoCalibrationProjectionError << std::endl;

  double leftProjectError2 = cvCalibrateCamera2(
                            &objectPointsLeft,
                            &imagePointsLeft,
                            &pointCountsLeft,
                            imageSize,
                            &outputIntrinsicMatrixLeft,
                            &outputDistortionCoefficientsLeft,
                            &outputRotationVectorsLeft,
                            &outputTranslationVectorsLeft,
                            CV_CALIB_FIX_INTRINSIC
                            );

  double rightProjectError2 = cvCalibrateCamera2(
                            &objectPointsRight,
                            &imagePointsRight,
                            &pointCountsRight,
                            imageSize,
                            &outputIntrinsicMatrixRight,
                            &outputDistortionCoefficientsRight,
                            &outputRotationVectorsRight,
                            &outputTranslationVectorsRight,
                            CV_CALIB_FIX_INTRINSIC
                            );

  std::cout << "Final extrinsic calibration gave re-projection errors of left=" << leftProjectError2 << ", right=" << rightProjectError2 << std::endl;

  return stereoCalibrationProjectionError;
}


//-----------------------------------------------------------------------------
void OutputCalibrationData(
    std::ostream& os,
    const CvMat& objectPoints,
    const CvMat& imagePoints,
    const CvMat& pointCounts,
    const CvMat& intrinsicMatrix,
    const CvMat& distortionCoeffs,
    const CvMat& rotationVectors,
    const CvMat& translationVectors,
    const float& projectionError,
    const int& sizeX,
    const int& sizeY,
    const int& cornersX,
    const int& cornersY,
    std::vector<std::string>& fileNames
    )
{
  int pointCount = cornersX * cornersY;
  int numberOfFilesUsed = fileNames.size();

  CvMat *extrinsicMatrix = cvCreateMat(4,4,CV_32FC1);
  CvMat *projectedImagePoints = cvCreateMat(numberOfFilesUsed*pointCount, 2, CV_32FC1);

  ProjectAllPoints(
      numberOfFilesUsed,
      pointCount,
      objectPoints,
      imagePoints,
      intrinsicMatrix,
      distortionCoeffs,
      rotationVectors,
      translationVectors,
      *projectedImagePoints
      );

  os.precision(10);
  os.width(10);

  os << "Intrinsic matrix" << std::endl;
  os << CV_MAT_ELEM(intrinsicMatrix, float, 0, 0) << ", " << CV_MAT_ELEM(intrinsicMatrix, float, 0, 1) << ", " << CV_MAT_ELEM(intrinsicMatrix, float, 0, 2) << std::endl;
  os << CV_MAT_ELEM(intrinsicMatrix, float, 1, 0) << ", " << CV_MAT_ELEM(intrinsicMatrix, float, 1, 1) << ", " << CV_MAT_ELEM(intrinsicMatrix, float, 1, 2) << std::endl;
  os << CV_MAT_ELEM(intrinsicMatrix, float, 2, 0) << ", " << CV_MAT_ELEM(intrinsicMatrix, float, 2, 1) << ", " << CV_MAT_ELEM(intrinsicMatrix, float, 2, 2) << std::endl;

  os << "Distortion vector (k1, k2, p1, p2, k3)" << std::endl;
  os << CV_MAT_ELEM(distortionCoeffs, float, 0, 0) << ", " << CV_MAT_ELEM(distortionCoeffs, float, 1, 0) << ", " << CV_MAT_ELEM(distortionCoeffs, float, 2, 0) << ", " << CV_MAT_ELEM(distortionCoeffs, float, 3, 0) << ", " << CV_MAT_ELEM(distortionCoeffs, float, 4, 0) << std::endl;

  os << "projection error:" << projectionError << std::endl;
  os << "image size:" << sizeX << " " << sizeY << std::endl;
  os << "number of internal corners:" << cornersX << " " << cornersY << std::endl;

  os << "number of files used:" << numberOfFilesUsed << std::endl;
  os << "list of files used:" << std::endl;

  // Also output files actually used.
  for (unsigned int i = 0; i < fileNames.size(); i++)
  {
    os << fileNames[i] << std::endl;
  }

  // Also output points on a per filename basis
  os << "points per file" << std::endl;
  unsigned int numberOfPoints = CV_MAT_ELEM(pointCounts, int, 0, 0);
  for (unsigned int i = 0; i < fileNames.size(); i++)
  {
    os << fileNames[i] << std::endl;

    ExtractExtrinsicMatrixFromRotationAndTranslationVectors(
        rotationVectors,
        translationVectors,
        i,
        *extrinsicMatrix
        );

    os << "Extrinsic matrix" << std::endl;
    for (int a = 0; a < 4; a++)
    {
      for (int b = 0; b < 4; b++)
      {
        os << CV_MAT_ELEM(*extrinsicMatrix, float, a, b);
        if (b < 3)
        {
          os << ", ";
        }
      }
      os << std::endl;
    }

    for (unsigned int j = 0; j < numberOfPoints; j++)
    {
      os << CV_MAT_ELEM(objectPoints, float, i*numberOfPoints + j, 0) << ", " << CV_MAT_ELEM(objectPoints, float, i*numberOfPoints + j, 1) << ", " << CV_MAT_ELEM(objectPoints, float, i*numberOfPoints + j, 2) \
          << " projects to " << CV_MAT_ELEM(*projectedImagePoints, float, i*numberOfPoints + j, 0) << ", " << CV_MAT_ELEM(*projectedImagePoints, float, i*numberOfPoints + j, 1) \
          << " compared with " << CV_MAT_ELEM(imagePoints, float, i*numberOfPoints + j, 0) << ", " << CV_MAT_ELEM(imagePoints, float, i*numberOfPoints + j, 1) \
          << " detected in image " \
          << std::endl;
    }
  }

  cvReleaseMat(&extrinsicMatrix);
  cvReleaseMat(&projectedImagePoints);
}


//-----------------------------------------------------------------------------
void CorrectDistortionInImageFile(
    const std::string& inputFileName,
    const CvMat& intrinsicParams,
    const CvMat& distortionCoefficients,
    const std::string& outputFileName
    )
{
  IplImage *image = cvLoadImage(inputFileName.c_str());
  if (image == NULL)
  {
    throw std::logic_error("Failed to load image");
  }
  CorrectDistortionInSingleImage(intrinsicParams, distortionCoefficients, *image);
  cvSaveImage(outputFileName.c_str(), image);
  cvReleaseImage(&image);
}


//-----------------------------------------------------------------------------
void CorrectDistortionInImageFile(
    const std::string& inputImageFileName,
    const std::string& inputIntrinsicsFileName,
    const std::string& inputDistortionCoefficientsFileName,
    const std::string& outputImageFileName
    )
{
  CvMat *intrinsic = (CvMat*)cvLoad(inputIntrinsicsFileName.c_str());
  if (intrinsic == NULL)
  {
    throw std::logic_error("Failed to load amera intrinsic params");
  }

  CvMat *distortion = (CvMat*)cvLoad(inputDistortionCoefficientsFileName.c_str());
  if (distortion == NULL)
  {
    throw std::logic_error("Failed to load camera distortion params");
  }

  CorrectDistortionInImageFile(inputImageFileName, *intrinsic, *distortion, outputImageFileName);

  cvReleaseMat(&intrinsic);
  cvReleaseMat(&distortion);
}


//-----------------------------------------------------------------------------
void CorrectDistortionInSingleImage(
    const CvMat& intrinsicParams,
    const CvMat& distortionCoefficients,
    IplImage &image
    )
{
  IplImage *mapX = cvCreateImage(cvGetSize(&image), IPL_DEPTH_32F, 1);
  IplImage *mapY = cvCreateImage(cvGetSize(&image), IPL_DEPTH_32F, 1);

  cvInitUndistortMap(&intrinsicParams, &distortionCoefficients, mapX, mapY);
  UndistortImageUsingDistortionMap(*mapX, *mapY, image);
}


//-----------------------------------------------------------------------------
void UndistortImageUsingDistortionMap(
    const IplImage &mapX,
    const IplImage &mapY,
    IplImage &image
    )
{
  IplImage *tmp = cvCloneImage(&image);
  ApplyDistortionCorrectionMap(mapX, mapY, *tmp, image);
  cvReleaseImage(&tmp);
}


//-----------------------------------------------------------------------------
void ApplyDistortionCorrectionMap(
    const IplImage &mapX,
    const IplImage &mapY,
    const IplImage &inputImage,
    IplImage &outputImage
    )
{
  cvRemap(&inputImage, &outputImage, &mapX, &mapY);
}

//-----------------------------------------------------------------------------
} // end namespace
