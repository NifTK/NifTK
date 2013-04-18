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
#include <fstream>
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
                             const double& squareSizeInMillimetres,
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
        CV_MAT_ELEM(*objectPoints, float, j, 0) = (k/numberCornersWidth)*squareSizeInMillimetres;
        CV_MAT_ELEM(*objectPoints, float, j, 1) = (k%numberCornersWidth)*squareSizeInMillimetres;
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
void ComputeRightToLeftTransformations(
    const CvMat& rotationVectorsLeft,
    const CvMat& translationVectorsLeft,
    const CvMat& rotationVectorsRight,
    const CvMat& translationVectorsRight,
    const CvMat& rotationVectorsRightToLeft,
    const CvMat& translationVectorsRightToLeft
    )
{
  int numberOfMatrices = rotationVectorsLeft.rows;

  if (translationVectorsLeft.rows != numberOfMatrices
      || rotationVectorsRight.rows != numberOfMatrices
      || translationVectorsRight.rows != numberOfMatrices
      || rotationVectorsRightToLeft.rows != numberOfMatrices
      || translationVectorsRightToLeft.rows != numberOfMatrices
      )
  {
    throw std::logic_error("Inconsistent number of rows in supplied matrices!");
  }

  CvMat *leftCameraTransform = cvCreateMat(4, 4, CV_32FC1);
  CvMat *rightCameraTransform = cvCreateMat(4, 4, CV_32FC1);
  CvMat *rightCameraTransformInverted = cvCreateMat(4, 4, CV_32FC1);
  CvMat *rightToLeftCameraTransform = cvCreateMat(4, 4, CV_32FC1);
  CvMat *rotationMatrix = cvCreateMat(3, 3, CV_32FC1);
  CvMat *rotationVector = cvCreateMat(1, 3, CV_32FC1);

  cvSetZero(rotationVector);
  cvSetZero(rotationMatrix);

  for (int i = 0; i < numberOfMatrices; i++)
  {
    ExtractExtrinsicMatrixFromRotationAndTranslationVectors(rotationVectorsLeft, translationVectorsLeft, i, *leftCameraTransform);
    ExtractExtrinsicMatrixFromRotationAndTranslationVectors(rotationVectorsRight, translationVectorsRight, i, *rightCameraTransform);
    cvInvert(rightCameraTransform, rightCameraTransformInverted);
    cvGEMM(leftCameraTransform, rightCameraTransformInverted, 1, NULL, 0, rightToLeftCameraTransform);

    for (int j = 0; j < 3; j++)
    {
      for (int k = 0; k < 3; k++)
      {
        CV_MAT_ELEM(*rotationMatrix, float, j, k) = CV_MAT_ELEM(*rightToLeftCameraTransform, float, j, k);
      }
    }
    cvRodrigues2(rotationMatrix, rotationVector);

    // Write output
    for (int j = 0; j < 3; j++)
    {
      CV_MAT_ELEM(rotationVectorsRightToLeft, float, i, j) = CV_MAT_ELEM(*rotationVector, float, 0, j);
      CV_MAT_ELEM(translationVectorsRightToLeft, float, i, j) = CV_MAT_ELEM(*rightToLeftCameraTransform, float, j, 3);
    }
  }

  cvReleaseMat(&leftCameraTransform);
  cvReleaseMat(&rightCameraTransform);
  cvReleaseMat(&rightCameraTransformInverted);
  cvReleaseMat(&rightToLeftCameraTransform);
  cvReleaseMat(&rotationMatrix);
  cvReleaseMat(&rotationVector);
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
      cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-6), // where cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, 1e-6) is the default.
      CV_CALIB_FIX_INTRINSIC // Use the initial guess, but feel free to optimise it.
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
  CvMat *modelPointInputHomogeneous = cvCreateMat(4,1,CV_32FC1);
  CvMat *modelPointOutputHomogeneous = cvCreateMat(4,1,CV_32FC1);
  CvMat *extrinsicRotationVector = cvCreateMat(1,3,CV_32FC1);
  CvMat *extrinsicTranslationVector = cvCreateMat(1,3,CV_32FC1);
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

    CV_MAT_ELEM(*extrinsicRotationVector, float, 0, 0) = CV_MAT_ELEM(rotationVectors, float, i, 0);
    CV_MAT_ELEM(*extrinsicRotationVector, float, 0, 1) = CV_MAT_ELEM(rotationVectors, float, i, 1);
    CV_MAT_ELEM(*extrinsicRotationVector, float, 0, 2) = CV_MAT_ELEM(rotationVectors, float, i, 2);

    CV_MAT_ELEM(*extrinsicTranslationVector, float, 0, 0) = CV_MAT_ELEM(translationVectors, float, i, 0);
    CV_MAT_ELEM(*extrinsicTranslationVector, float, 0, 1) = CV_MAT_ELEM(translationVectors, float, i, 1);
    CV_MAT_ELEM(*extrinsicTranslationVector, float, 0, 2) = CV_MAT_ELEM(translationVectors, float, i, 2);

    ExtractExtrinsicMatrixFromRotationAndTranslationVectors(
        rotationVectors,
        translationVectors,
        i,
        *extrinsicMatrix
        );

    cvSave(std::string(fileNames[i] + ".extrinsic.xml").c_str(), extrinsicMatrix);
    cvSave(std::string(fileNames[i] + ".extrinsic.rot.xml").c_str(), extrinsicRotationVector);
    cvSave(std::string(fileNames[i] + ".extrinsic.trans.xml").c_str(), extrinsicTranslationVector);

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
      CV_MAT_ELEM(*modelPointInputHomogeneous, float, 0 ,0) = CV_MAT_ELEM(objectPoints, float, i*numberOfPoints + j, 0);
      CV_MAT_ELEM(*modelPointInputHomogeneous, float, 1, 0) = CV_MAT_ELEM(objectPoints, float, i*numberOfPoints + j, 1);
      CV_MAT_ELEM(*modelPointInputHomogeneous, float, 2, 0) = CV_MAT_ELEM(objectPoints, float, i*numberOfPoints + j, 2);
      CV_MAT_ELEM(*modelPointInputHomogeneous, float, 3, 0) = 1;

      cvGEMM(extrinsicMatrix, modelPointInputHomogeneous, 1, NULL, 0, modelPointOutputHomogeneous);

      os << CV_MAT_ELEM(objectPoints, float, i*numberOfPoints + j, 0) << ", " << CV_MAT_ELEM(objectPoints, float, i*numberOfPoints + j, 1) << ", " << CV_MAT_ELEM(objectPoints, float, i*numberOfPoints + j, 2) \
          << " transforms to " << CV_MAT_ELEM(*modelPointOutputHomogeneous, float, 0 ,0) << ", " << CV_MAT_ELEM(*modelPointOutputHomogeneous, float, 1 ,0) << ", " << CV_MAT_ELEM(*modelPointOutputHomogeneous, float, 2 ,0) \
          << " and then projects to " << CV_MAT_ELEM(*projectedImagePoints, float, i*numberOfPoints + j, 0) << ", " << CV_MAT_ELEM(*projectedImagePoints, float, i*numberOfPoints + j, 1) \
          << " compared with " << CV_MAT_ELEM(imagePoints, float, i*numberOfPoints + j, 0) << ", " << CV_MAT_ELEM(imagePoints, float, i*numberOfPoints + j, 1) \
          << " detected in image " \
          << std::endl;
    }
  }

  cvReleaseMat(&extrinsicMatrix);
  cvReleaseMat(&modelPointInputHomogeneous);
  cvReleaseMat(&modelPointOutputHomogeneous);
  cvReleaseMat(&extrinsicRotationVector);
  cvReleaseMat(&extrinsicTranslationVector);
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
    throw std::logic_error("Failed to load camera intrinsic params");
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
void Project3DModelPositionsToStereo2D(
    const CvMat& modelPointsIn3D,
    const CvMat& leftCameraIntrinsic,
    const CvMat& leftCameraDistortion,
    const CvMat& leftCameraRotationVector,
    const CvMat& leftCameraTranslationVector,
    const CvMat& rightCameraIntrinsic,
    const CvMat& rightCameraDistortion,
    const CvMat& rightToLeftRotationMatrix,
    const CvMat& rightToLeftTranslationVector,
    CvMat& output2DPointsLeft,
    CvMat& output2DPointsRight
    )
{

  // NOTE: modelPointsIn3D should be [Nx3]. i.e. N rows, 3 columns.
  cvProjectPoints2(
      &modelPointsIn3D,
      &leftCameraRotationVector,
      &leftCameraTranslationVector,
      &leftCameraIntrinsic,
      &leftCameraDistortion,
      &output2DPointsLeft
      );

  CvMat *leftCameraRotationMatrix = cvCreateMat(3, 3, CV_32FC1);
  cvRodrigues2(&leftCameraRotationVector, leftCameraRotationMatrix);

  CvMat *modelPointsIn3DInLeftCameraSpace = cvCreateMat(modelPointsIn3D.cols, modelPointsIn3D.rows, CV_32FC1);   // So, [3xN] matrix.
  cvGEMM(leftCameraRotationMatrix, &modelPointsIn3D, 1, NULL, 0, modelPointsIn3DInLeftCameraSpace, CV_GEMM_B_T); // ie. [3x3][Nx3]^T = [3xN].

  // This bit just to add the translation on, to get 3D model points, into 3D camera coordinates for left camera.
  for (int i = 0; i < modelPointsIn3D.rows; i++)
  {
    CV_MAT_ELEM(*modelPointsIn3DInLeftCameraSpace, float, 0, i) = CV_MAT_ELEM(*modelPointsIn3DInLeftCameraSpace, float, 0, i) + CV_MAT_ELEM(leftCameraTranslationVector, float, 0, 0);
    CV_MAT_ELEM(*modelPointsIn3DInLeftCameraSpace, float, 1, i) = CV_MAT_ELEM(*modelPointsIn3DInLeftCameraSpace, float, 1, i) + CV_MAT_ELEM(leftCameraTranslationVector, float, 0, 1);
    CV_MAT_ELEM(*modelPointsIn3DInLeftCameraSpace, float, 2, i) = CV_MAT_ELEM(*modelPointsIn3DInLeftCameraSpace, float, 2, i) + CV_MAT_ELEM(leftCameraTranslationVector, float, 0, 2);
  }

  CvMat *modelPointsIn3DInRightCameraSpace = cvCreateMat(modelPointsIn3D.cols, modelPointsIn3D.rows, CV_32FC1);        // So, [3xN] matrix.
  cvGEMM(&rightToLeftRotationMatrix, modelPointsIn3DInLeftCameraSpace, 1, NULL, 0, modelPointsIn3DInRightCameraSpace); // ie. [3x3][3xN] = [3xN]

  // Add translation again, to get 3D model points into 3D camera coordinates for right camera.
  for (int i = 0; i < modelPointsIn3D.rows; i++)
  {
    CV_MAT_ELEM(*modelPointsIn3DInRightCameraSpace, float, 0, i) = CV_MAT_ELEM(*modelPointsIn3DInRightCameraSpace, float, 0, i) + CV_MAT_ELEM(rightToLeftTranslationVector, float, 0, 0);
    CV_MAT_ELEM(*modelPointsIn3DInRightCameraSpace, float, 1, i) = CV_MAT_ELEM(*modelPointsIn3DInRightCameraSpace, float, 1, i) + CV_MAT_ELEM(rightToLeftTranslationVector, float, 1, 0);
    CV_MAT_ELEM(*modelPointsIn3DInRightCameraSpace, float, 2, i) = CV_MAT_ELEM(*modelPointsIn3DInRightCameraSpace, float, 2, i) + CV_MAT_ELEM(rightToLeftTranslationVector, float, 2, 0);
  }

  // Now project those points to 2D
  CvMat *rightCameraRotationMatrix = cvCreateMat(3, 3, CV_32FC1);
  CvMat *rightCameraRotationVector = cvCreateMat(1, 3, CV_32FC1);

  cvSetIdentity(rightCameraRotationMatrix);
  cvRodrigues2(rightCameraRotationMatrix, rightCameraRotationVector);

  CvMat *rightCameraTranslationVector = cvCreateMat(1, 3, CV_32FC1);
  cvSetZero(rightCameraTranslationVector);

  CvMat *modelPointsIn3DInRightCameraSpaceTransposed = cvCreateMat(modelPointsIn3D.rows, modelPointsIn3D.cols, CV_32FC1);
  cvTranspose(modelPointsIn3DInRightCameraSpace, modelPointsIn3DInRightCameraSpaceTransposed);

  cvProjectPoints2(
      modelPointsIn3DInRightCameraSpaceTransposed,
      rightCameraRotationVector,
      rightCameraTranslationVector,
      &rightCameraIntrinsic,
      &rightCameraDistortion,
      &output2DPointsRight
      );

  cvReleaseMat(&leftCameraRotationMatrix);
  cvReleaseMat(&modelPointsIn3DInLeftCameraSpace);
  cvReleaseMat(&modelPointsIn3DInRightCameraSpace);
  cvReleaseMat(&rightCameraRotationMatrix);
  cvReleaseMat(&rightCameraRotationVector);
  cvReleaseMat(&rightCameraTranslationVector);
  cvReleaseMat(&modelPointsIn3DInRightCameraSpaceTransposed);
}


//-----------------------------------------------------------------------------
std::vector<int> ProjectVisible3DWorldPointsToStereo2D(
    const CvMat& leftCameraWorldPointsIn3D,
    const CvMat& leftCameraWorldNormalsIn3D,
    const CvMat& leftCameraPositionToFocalPointUnitVector,
    const CvMat& leftCameraIntrinsic,
    const CvMat& leftCameraDistortion,
    const CvMat& rightCameraIntrinsic,
    const CvMat& rightCameraDistortion,
    const CvMat& rightToLeftRotationMatrix,
    const CvMat& rightToLeftTranslationVector,
    CvMat*& outputLeftCameraWorldPointsIn3D,
    CvMat*& outputLeftCameraWorldNormalsIn3D,
    CvMat*& output2DPointsLeft,
    CvMat*& output2DPointsRight
    )
{
  if (   outputLeftCameraWorldPointsIn3D != NULL
      || outputLeftCameraWorldNormalsIn3D != NULL
      || output2DPointsLeft != NULL
      || output2DPointsRight != NULL
      )
  {
    throw std::logic_error("Output pointers should be NULL, as this method creates new matrices");
  }

  int numberOfInputPoints = leftCameraWorldPointsIn3D.rows;
  int numberOfOutputPoints = 0;

  std::vector<int> validPoints;

  for (int i = 0; i < numberOfInputPoints; i++)
  {
    double cosAngleBetweenTwoVectors =
          CV_MAT_ELEM(leftCameraWorldNormalsIn3D, float, i, 0) * CV_MAT_ELEM(leftCameraPositionToFocalPointUnitVector, float, 0, 0)
        + CV_MAT_ELEM(leftCameraWorldNormalsIn3D, float, i, 1) * CV_MAT_ELEM(leftCameraPositionToFocalPointUnitVector, float, 0, 1)
        + CV_MAT_ELEM(leftCameraWorldNormalsIn3D, float, i, 2) * CV_MAT_ELEM(leftCameraPositionToFocalPointUnitVector, float, 0, 2)
        ;

    if (cosAngleBetweenTwoVectors < -0.1)
    {
      validPoints.push_back(i);
    }
  }

  numberOfOutputPoints = validPoints.size();

  if (numberOfOutputPoints > 0)
  {
    outputLeftCameraWorldPointsIn3D = cvCreateMat(numberOfOutputPoints, 3, CV_32FC1);
    outputLeftCameraWorldNormalsIn3D = cvCreateMat(numberOfOutputPoints, 3, CV_32FC1);
    output2DPointsLeft = cvCreateMat(numberOfOutputPoints, 2, CV_32FC1);
    output2DPointsRight = cvCreateMat(numberOfOutputPoints, 2, CV_32FC1);

    // Copy valid points to output arrays.
    for (unsigned int i = 0; i < validPoints.size(); i++)
    {
      for (unsigned int j = 0; j < 3; j++)
      {
        CV_MAT_ELEM(*outputLeftCameraWorldPointsIn3D, float, i, j) = CV_MAT_ELEM(leftCameraWorldPointsIn3D, float, validPoints[i], j);
        CV_MAT_ELEM(*outputLeftCameraWorldNormalsIn3D, float, i, j) = CV_MAT_ELEM(leftCameraWorldNormalsIn3D, float, validPoints[i], j);
      }
    }

    // Input points are already in world coordinates,
    // so we don't need left camera extrinsic parameters.
    CvMat *leftCameraRotationMatrix = cvCreateMat(3, 3, CV_32FC1);
    CvMat *leftCameraRotationVector = cvCreateMat(1, 3, CV_32FC1);

    cvSetIdentity(leftCameraRotationMatrix);
    cvRodrigues2(leftCameraRotationMatrix, leftCameraRotationVector);

    CvMat *leftCameraTranslationVector = cvCreateMat(1, 3, CV_32FC1);
    cvSetZero(leftCameraTranslationVector);

    // Now do point projection only on valid points
    Project3DModelPositionsToStereo2D(
        *outputLeftCameraWorldPointsIn3D,
        leftCameraIntrinsic,
        leftCameraDistortion,
        *leftCameraRotationVector,
        *leftCameraTranslationVector,
        rightCameraIntrinsic,
        rightCameraDistortion,
        rightToLeftRotationMatrix,
        rightToLeftTranslationVector,
        *output2DPointsLeft,
        *output2DPointsRight
        );

    // Tidy up, but DONT delete the output matrices.
    cvReleaseMat(&leftCameraRotationMatrix);
    cvReleaseMat(&leftCameraRotationVector);
    cvReleaseMat(&leftCameraTranslationVector);
  }

  return validPoints;
}


//-----------------------------------------------------------------------------
void UndistortPoints(const cv::Mat& inputObservedPointsNx2,
    const cv::Mat& cameraIntrinsics,
    const cv::Mat& cameraDistortionParams,
    cv::Mat& outputIdealPointsNx2
    )
{
  assert(inputObservedPointsNx2.rows == outputIdealPointsNx2.rows);
  assert(inputObservedPointsNx2.cols == outputIdealPointsNx2.cols);

  int numberOfPoints = inputObservedPointsNx2.rows;

  std::vector<cv::Point2f> inputPoints;
  inputPoints.resize(numberOfPoints);

  std::vector<cv::Point2f> outputPoints;
  outputPoints.resize(numberOfPoints);

  for (int i = 0; i < numberOfPoints; i++)
  {
    inputPoints[i].x = inputObservedPointsNx2.at<float>(i,0);
    inputPoints[i].y = inputObservedPointsNx2.at<float>(i,1);
  }

  UndistortPoints(inputPoints, cameraIntrinsics, cameraDistortionParams, outputPoints);

  for (int i = 0; i < numberOfPoints; i++)
  {
    outputIdealPointsNx2.at<float>(i,0) = outputPoints[i].x;
    outputIdealPointsNx2.at<float>(i,1) = outputPoints[i].y;
  }
}


//-----------------------------------------------------------------------------
void UndistortPoints(const std::vector<cv::Point2f>& inputPoints,
    const cv::Mat& cameraIntrinsics,
    const cv::Mat& cameraDistortionParams,
    std::vector<cv::Point2f>& outputPoints
    )
{
  cv::undistortPoints(inputPoints, outputPoints, cameraIntrinsics, cameraDistortionParams, cv::noArray(), cameraIntrinsics);
}


//-----------------------------------------------------------------------------
std::vector< cv::Point3f > TriangulatePointPairs(
    const std::vector< std::pair<cv::Point2f, cv::Point2f> >& inputUndistortedPoints,
    const cv::Mat& leftCameraIntrinsicParams,
    const cv::Mat& rightCameraIntrinsicParams,
    const cv::Mat& rightToLeftRotationVector,
    const cv::Mat& rightToLeftTranslationVector
    )
{
  int numberOfPoints = inputUndistortedPoints.size();
  std::vector< cv::Point3f > outputPoints;

  cv::Mat K1      = cv::Mat(3, 3, CV_64FC1);
  cv::Mat K2      = cv::Mat(3, 3, CV_64FC1);
  cv::Mat R2LRot  = cv::Mat(3, 3, CV_64FC1);
  cv::Mat R2LInv  = cv::Mat(3, 3, CV_64FC1);
  cv::Mat K1Inv   = cv::Mat(3, 3, CV_64FC1);
  cv::Mat K2Inv   = cv::Mat(3, 3, CV_64FC1);

  cv::Rodrigues(rightToLeftRotationVector, R2LRot);
  R2LInv = R2LRot.inv();

  // Copy data into cv::Mat data types.
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      K1.at<double>(i,j) = leftCameraIntrinsicParams.at<double>(i,j);
      K2.at<double>(i,j) = rightCameraIntrinsicParams.at<double>(i,j);
    }
  }

  // We invert the intrinsic params, so we can convert from pixels to normalised image coordinates.
  K1Inv = K1.inv();
  K2Inv = K2.inv();

  // Set up some working matrices...
  cv::Mat p1                = cv::Mat(3, 1, CV_64FC1);
  cv::Mat p2                = cv::Mat(3, 1, CV_64FC1);
  cv::Mat p1normalised      = cv::Mat(3, 1, CV_64FC1);
  cv::Mat p2normalised      = cv::Mat(3, 1, CV_64FC1);
  cv::Mat rhsRay            = cv::Mat(3, 1, CV_64FC1);
  cv::Mat rhsRayTransformed = cv::Mat(3, 1, CV_64FC1);

  // Line from left camera = P0 + \lambda_1 u;
  cv::Point3d P0;
  cv::Point3d u;

  // Line from right camera = Q0 + \lambda_2 v;
  cv::Point3d Q0;
  cv::Point3d v;

  double UNorm, VNorm;

  // For each point...
  for (int i = 0; i < numberOfPoints; i++)
  {
    p1.at<double>(0,0) = inputUndistortedPoints[i].first.x;
    p1.at<double>(1,0) = inputUndistortedPoints[i].first.y;
    p1.at<double>(2,0) = 1;

    p2.at<double>(0,0) = inputUndistortedPoints[i].second.x;
    p2.at<double>(1,0) = inputUndistortedPoints[i].second.y;
    p2.at<double>(2,0) = 1;

    // Converting to normalised image points.
    p1normalised = K1Inv * p1;
    p2normalised = K2Inv * p2;

    // Origin in LH camera, by definition is 0,0,0.
    P0.x = 0;
    P0.y = 0;
    P0.z = 0;

    // Create unit vector along left hand camera line.
    UNorm = sqrt(p1normalised.at<double>(0,0)*p1normalised.at<double>(0,0)
               + p1normalised.at<double>(1,0)*p1normalised.at<double>(1,0)
               + p1normalised.at<double>(2,0)*p1normalised.at<double>(2,0)
               );
    u.x = p1normalised.at<double>(0,0)/UNorm;
    u.y = p1normalised.at<double>(1,0)/UNorm;
    u.z = p1normalised.at<double>(2,0)/UNorm;

    // Calculate unit vector in right hand coordinate system.
    VNorm = sqrt(p2normalised.at<double>(0,0)*p2normalised.at<double>(0,0)
               + p2normalised.at<double>(1,0)*p2normalised.at<double>(1,0)
               + p2normalised.at<double>(2,0)*p2normalised.at<double>(2,0));

    rhsRay.at<double>(0,0) = p2normalised.at<double>(0,0) / VNorm;
    rhsRay.at<double>(1,0) = p2normalised.at<double>(1,0) / VNorm;
    rhsRay.at<double>(2,0) = p2normalised.at<double>(2,0) / VNorm;

    // Rotate unit vector by rotation matrix between left and right camera.
    rhsRayTransformed = R2LRot * rhsRay;

    // Origin of RH camera, in LH normalised coordinates.
    Q0.x = rightToLeftTranslationVector.at<double>(0,0);
    Q0.y = rightToLeftTranslationVector.at<double>(0,1);
    Q0.z = rightToLeftTranslationVector.at<double>(0,2);

    // Create unit vector along right hand camera line, but in LH coordinate frame.
    v.x = rhsRayTransformed.at<double>(0,0);
    v.y = rhsRayTransformed.at<double>(1,0);
    v.z = rhsRayTransformed.at<double>(2,0);

    // Method 1. Solve for shortest line joining two rays, then get midpoint.
    // Taken from: http://geomalgorithms.com/a07-_distance.html

    double sc, tc, a, b, c, d, e;

    cv::Point3d Psc;
    cv::Point3d Qtc;
    cv::Point3d W0;
    cv::Point3d midPoint;

    // Difference of two origins
    W0.x = P0.x - Q0.x;
    W0.y = P0.y - Q0.y;
    W0.z = P0.z - Q0.z;

    a = u.x*u.x + u.y*u.y + u.z*u.z;
    b = u.x*v.x + u.y*v.y + u.z*v.z;
    c = v.x*v.x + v.y*v.y + v.z*v.z;
    d = u.x*W0.x + u.y*W0.y + u.z*W0.z;
    e = v.x*W0.x + v.y*W0.y + v.z*W0.z;

    sc = (b*e - c*d) / (a*c - b*b);
    tc = (a*e - b*d) / (a*c - b*b);

    if (sc > 0 && tc > 0)
    {
      Psc.x = P0.x + sc*u.x;
      Psc.y = P0.y + sc*u.y;
      Psc.z = P0.z + sc*u.z;

      Qtc.x = Q0.x + tc*v.x;
      Qtc.y = Q0.y + tc*v.y;
      Qtc.z = Q0.z + tc*v.z;

      midPoint.x = (Psc.x + Qtc.x)/2.0;
      midPoint.y = (Psc.y + Qtc.y)/2.0;
      midPoint.z = (Psc.z + Qtc.z)/2.0;

      outputPoints.push_back(midPoint);

      /*
      std::cerr << "Matt, l=(" << inputUndistortedPoints[i].first.x \
          << ", " << inputUndistortedPoints[i].first.y \
          << "), r=(" << inputUndistortedPoints[i].second.x \
          << ", " << inputUndistortedPoints[i].second.y \
          << "), 3D=" << midPoint.x \
          << ", " << midPoint.y \
          << ", " << midPoint.z \
          << std::endl;
      */
    }

    // Method 2. SVD it - I tested this, and the numbers came out the same.
    /*
    cv::Matx32d A(u.x, -v.x,
                  u.y, -v.y,
                  u.z, -v.z
                 );

    cv::Matx31d B(Q0.x - P0x,
                  Q0.y - P0y,
                  Q0.z - P0z
                 );

    cv::Mat_<double> X;
    cv::solve(A,B,X,cv::DECOMP_SVD);
    */
  }
  return outputPoints;
}


//-----------------------------------------------------------------------------
void TriangulatePointPairs(
    const CvMat& leftCameraUndistortedImagePoints,
    const CvMat& rightCameraUndistortedImagePoints,
    const CvMat& leftCameraIntrinsicParams,
    const CvMat& leftCameraRotationVector,
    const CvMat& leftCameraTranslationVector,
    const CvMat& rightCameraIntrinsicParams,
    const CvMat& rightCameraRotationVector,
    const CvMat& rightCameraTranslationVector,
    CvMat& output3DPoints
    )
{
  int numberOfPoints = leftCameraUndistortedImagePoints.rows;

  cv::Mat K1(&leftCameraIntrinsicParams,false);
  cv::Mat R1(&leftCameraRotationVector, false);
  cv::Mat T1(&leftCameraTranslationVector, false);
  cv::Mat K2(&rightCameraIntrinsicParams, false);
  cv::Mat R2(&rightCameraRotationVector, false);
  cv::Mat T2(&rightCameraTranslationVector, false);

  std::vector< std::pair<cv::Point2f, cv::Point2f> > inputPairs;
  std::vector< cv::Point3f > outputPoints;

  cv::Point2f leftPoint;
  cv::Point2f rightPoint;
  cv::Point3f reconstructedPoint;

  for (int i = 0; i < numberOfPoints; i++)
  {
    leftPoint.x = CV_MAT_ELEM(leftCameraUndistortedImagePoints, float, i, 0);
    leftPoint.y = CV_MAT_ELEM(leftCameraUndistortedImagePoints, float, i, 0);
    rightPoint.x = CV_MAT_ELEM(rightCameraUndistortedImagePoints, float, i, 0);
    rightPoint.y = CV_MAT_ELEM(rightCameraUndistortedImagePoints, float, i, 0);
  }

  inputPairs.push_back( std::pair<cv::Point2f, cv::Point2f>(leftPoint, rightPoint));

  // Call the other, more C++ like method.
  outputPoints = TriangulatePointPairs(
      inputPairs,
      K1, R1, T1,
      K2, R2, T2
      );

  // Now convert points back
  for (unsigned int i = 0; i < outputPoints.size(); i++)
  {
    reconstructedPoint = outputPoints[i];
    CV_MAT_ELEM(output3DPoints, float, i, 0) = reconstructedPoint.x;
    CV_MAT_ELEM(output3DPoints, float, i, 1) = reconstructedPoint.y;
    CV_MAT_ELEM(output3DPoints, float, i, 2) = reconstructedPoint.z;
  }
}


//-----------------------------------------------------------------------------
std::vector< cv::Point3f > TriangulatePointPairs(
    const std::vector< std::pair<cv::Point2f, cv::Point2f> >& inputUndistortedPoints,
    const cv::Mat& leftCameraIntrinsicParams,
    const cv::Mat& leftCameraRotationVector,
    const cv::Mat& leftCameraTranslationVector,
    const cv::Mat& rightCameraIntrinsicParams,
    const cv::Mat& rightCameraRotationVector,
    const cv::Mat& rightCameraTranslationVector
    )
{

  int numberOfPoints = inputUndistortedPoints.size();
  std::vector< cv::Point3f > outputPoints;

  cv::Mat K1    = cv::Mat(3, 3, CV_64FC1);
  cv::Mat K2    = cv::Mat(3, 3, CV_64FC1);
  cv::Mat K1Inv = cv::Mat(3, 3, CV_64FC1);
  cv::Mat K2Inv = cv::Mat(3, 3, CV_64FC1);
  cv::Mat R1    = cv::Mat(3, 3, CV_64FC1);
  cv::Mat R2    = cv::Mat(3, 3, CV_64FC1);
  cv::Mat E1    = cv::Mat(4, 4, CV_64FC1);
  cv::Mat E1Inv = cv::Mat(4, 4, CV_64FC1);
  cv::Mat E2    = cv::Mat(4, 4, CV_64FC1);
  cv::Mat L2R   = cv::Mat(4, 4, CV_64FC1);
  cv::Mat u1    = cv::Mat(3, 1, CV_64FC1);
  cv::Mat u2    = cv::Mat(3, 1, CV_64FC1);
  cv::Mat u1t   = cv::Mat(3, 1, CV_64FC1);
  cv::Mat u2t   = cv::Mat(3, 1, CV_64FC1);
  cv::Matx34d P1d, P2d;

  // Convert OpenCV stylee rotation vector to a rotation matrix.
  cv::Rodrigues(leftCameraRotationVector, R1);
  cv::Rodrigues(rightCameraRotationVector, R2);

  // Construct:
  // E1 = Object to Left Camera = Left Camera Extrinsics.
  // E2 = Object to Right Camera = Right Camera Extrinsics.
  // K1 = Copy of Left Camera intrinsics.
  // K2 = Copy of Right Camera intrinsics.
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      K1.at<double>(i,j) = leftCameraIntrinsicParams.at<double>(i,j);
      K2.at<double>(i,j) = rightCameraIntrinsicParams.at<double>(i,j);
      E1.at<double>(i,j) = R1.at<double>(i,j);
      E2.at<double>(i,j) = R2.at<double>(i,j);
    }
    E1.at<double>(i,3) = leftCameraTranslationVector.at<double>(0,i);
    E2.at<double>(i,3) = rightCameraTranslationVector.at<double>(0,i);
  }
  E1.at<double>(3,0) = 0;
  E1.at<double>(3,1) = 0;
  E1.at<double>(3,2) = 0;
  E1.at<double>(3,3) = 1;
  E2.at<double>(3,0) = 0;
  E2.at<double>(3,1) = 0;
  E2.at<double>(3,2) = 0;
  E2.at<double>(3,3) = 1;

  // We invert the intrinsic params, so we can convert from pixels to normalised image coordinates.
  K1Inv = K1.inv();
  K2Inv = K2.inv();

  // We want output coordinates relative to left camera.
  E1Inv = E1.inv();
  L2R = E2 * E1Inv;

  // Reading Prince 2012 Computer Vision, the projection matrix, is just the extrinsic parameters,
  // as our coordinates will be in a normalised camera space. P1 should be identity, so that
  // reconstructed coordinates are in Left Camera Space, to P2 should reflect a right to left transform.
  P1d(0,0) = 1; P1d(0,1) = 0; P1d(0,2) = 0; P1d(0,3) = 0;
  P1d(1,0) = 0; P1d(1,1) = 1; P1d(1,2) = 0; P1d(1,3) = 0;
  P1d(2,0) = 0; P1d(2,1) = 0; P1d(2,2) = 1; P1d(2,3) = 0;

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      P2d(i,j) = L2R.at<double>(i,j);
    }
  }

  cv::Point3d u1p, u2p;            // Normalised image coordinates. (i.e. relative to a principal point of zero, and in millimetres not pixels).
  cv::Point3d reconstructedPoint;  // the output 3D point, in reference frame of left camera.

  for (int i = 0; i < numberOfPoints; i++)
  {
    u1.at<double>(0,0) = inputUndistortedPoints[i].first.x;
    u1.at<double>(1,0) = inputUndistortedPoints[i].first.y;
    u1.at<double>(2,0) = 1;

    u2.at<double>(0,0) = inputUndistortedPoints[i].second.x;
    u2.at<double>(1,0) = inputUndistortedPoints[i].second.y;
    u2.at<double>(2,0) = 1;

    // Converting to normalised image points
    u1t = K1Inv * u1;
    u2t = K2Inv * u2;

    u1p.x = u1t.at<double>(0,0);
    u1p.y = u1t.at<double>(1,0);
    u1p.z = u1t.at<double>(2,0);

    u2p.x = u2t.at<double>(0,0);
    u2p.y = u2t.at<double>(1,0);
    u2p.z = u2t.at<double>(2,0);

    reconstructedPoint = IterativeTriangulatePoint(P1d, P2d, u1p, u2p);
    outputPoints.push_back(reconstructedPoint);
/*
    std::cout << "TriangulatePointPairs:l=(" << inputUndistortedPoints[i].first.x << ", " << inputUndistortedPoints[i].first.y << "), r=(" << inputUndistortedPoints[i].second.x << ", " << inputUndistortedPoints[i].second.y << "), 3D=" << reconstructedPoint.x << ", " << reconstructedPoint.y << ", " << reconstructedPoint.z << ")" << std::endl;
*/
  }

  return outputPoints;
}


//-----------------------------------------------------------------------------
cv::Mat_<double> TriangulatePoint(
    const cv::Matx34d& P1,
    const cv::Matx34d& P2,
    const cv::Point3d& u1,
    const cv::Point3d& u2,
    const double& w1,
    const double& w2
    )
{
  // Build matrix A for homogenous equation system Ax = 0
  // Assume X = (x,y,z,1), for Linear-LS method
  // Which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
  cv::Matx43d A((u1.x*P1(2,0)-P1(0,0))/w1, (u1.x*P1(2,1)-P1(0,1))/w1, (u1.x*P1(2,2)-P1(0,2))/w1,
                (u1.y*P1(2,0)-P1(1,0))/w1, (u1.y*P1(2,1)-P1(1,1))/w1, (u1.y*P1(2,2)-P1(1,2))/w1,
                (u2.x*P2(2,0)-P2(0,0))/w2, (u2.x*P2(2,1)-P2(0,1))/w2, (u2.x*P2(2,2)-P2(0,2))/w2,
                (u2.y*P2(2,0)-P2(1,0))/w2, (u2.y*P2(2,1)-P2(1,1))/w2, (u2.y*P2(2,2)-P2(1,2))/w2
               );


  cv::Matx41d B(-(u1.x*P1(2,3) -P1(0,3))/w1,
                -(u1.y*P1(2,3) -P1(1,3))/w1,
                -(u2.x*P2(2,3) -P2(0,3))/w2,
                -(u2.y*P2(2,3) -P2(1,3))/w2
               );

  cv::Mat_<double> X;
  cv::solve(A,B,X,cv::DECOMP_SVD);

  return X;
}


//-----------------------------------------------------------------------------
cv::Point3d IterativeTriangulatePoint(
    const cv::Matx34d& P1,
    const cv::Matx34d& P2,
    const cv::Point3d& u1,
    const cv::Point3d& u2
    )
{
  double epsilon = 0.00000000001;
  double w1 = 1, w2 = 1;
  cv::Mat_<double> X(4,1);

  for (int i=0; i<10; i++) // Hartley suggests 10 iterations at most
  {
    cv::Mat_<double> X_ = TriangulatePoint(P1,P2,u1,u2,w1,w2);
    X(0) = X_(0);
    X(1) = X_(1);
    X(2) = X_(2);
    X(3) = 1.0;

    double p2x1 = cv::Mat_<double>(cv::Mat_<double>(P1).row(2)*X)(0);
    double p2x2 = cv::Mat_<double>(cv::Mat_<double>(P2).row(2)*X)(0);

    if(fabs(w1 - p2x1) <= epsilon && fabs(w2 - p2x2) <= epsilon)
      break;

    w1 = p2x1;
    w2 = p2x2;
  }

  cv::Point3d result;
  result.x = X(0);
  result.y = X(1);
  result.z = X(2);

  return result;
}

//-----------------------------------------------------------------------------
std::vector<cv::Mat> LoadMatricesFromDirectory (const std::string& fullDirectoryName)
{
  std::vector<std::string> files = niftk::GetFilesInDirectory(fullDirectoryName);
  std::sort(files.begin(),files.end());
  std::vector<cv::Mat> myMatrices;

  if (files.size() > 0)
  {
    for(unsigned int i = 0; i < files.size();i++)
    {
      cv::Mat Matrix = cvCreateMat(4,4,CV_64FC1);
      std::ifstream fin(files[i].c_str());
      for ( int row = 0 ; row < 4 ; row ++ ) 
      {
        for ( int col = 0 ; col < 4 ; col ++ ) 
        {
          fin >> Matrix.at<double>(row,col);
        }
      }
      myMatrices.push_back(Matrix);
    }
  }
  else
  {
    throw std::logic_error("No files found in directory!");
  }

  if (myMatrices.size() == 0)
  {
    throw std::logic_error("No Matrices found in directory!");
  }
  std::cout << "Loaded " << myMatrices.size() << " Matrices from " << fullDirectoryName << std::endl;
  return myMatrices;
}
//-----------------------------------------------------------------------------
std::vector<cv::Mat> LoadOpenCVMatricesFromDirectory (const std::string& fullDirectoryName)
{
  std::vector<std::string> files = niftk::GetFilesInDirectory(fullDirectoryName);
  std::sort(files.begin(),files.end());
  std::vector<cv::Mat> myMatrices;

  if (files.size() > 0)
  {
    for(unsigned int i = 0; i < files.size();i++)
    {
      if ( niftk::FilenameHasPrefixAndExtension(files[i],"",".extrinsic.xml") )
      {
        cv::Mat Extrinsic = (cv::Mat)cvLoadImage(files[i].c_str());
        if (Extrinsic.rows != 4 )
        {
          throw std::logic_error("Failed to load camera intrinsic params");
        }
        else
        {
          myMatrices.push_back(Extrinsic);
          std::cout << "Loaded: " << Extrinsic << std::endl << "From " << files[i] << std::endl;
        }
      }
    }
  }
  else
  {
    throw std::logic_error("No files found in directory!");
  }

  if (myMatrices.size() == 0)
  {
    throw std::logic_error("No Matrices found in directory!");
  }
  std::cout << "Loaded " << myMatrices.size() << " Matrices from " << fullDirectoryName << std::endl;
  return myMatrices;
}

//------------------------------------------------------------------------------
void LoadResult (const std::string& FileName, cv::Mat& result, 
    std::vector<double>& residuals)
{
  std::ifstream fin(FileName.c_str());
  double temp;
  fin >> temp;
  residuals.push_back(temp);
  fin >> temp;
  residuals.push_back(temp);
  for ( int row = 0 ; row < 4 ; row ++ ) 
  {
    for ( int col = 0 ; col < 4 ; col ++ ) 
    {
      fin >> result.at<double>(row,col);
    }
  }

}


//-----------------------------------------------------------------------------
std::vector<cv::Mat> LoadMatricesFromExtrinsicFile (const std::string& fullFileName)
{

  std::vector<cv::Mat> myMatrices;
  std::ifstream fin(fullFileName.c_str());

  cv::Mat RotationVector = cvCreateMat(3,1,CV_64FC1);
  cv::Mat TranslationVector = cvCreateMat(3,1,CV_64FC1);
  double temp_d[6];
  while ( fin >> temp_d[0] >> temp_d[1] >> temp_d[2] >> temp_d[3] >> temp_d[4] >> temp_d[5] )
  {
    RotationVector.at<double>(0,0) = temp_d[0];
    RotationVector.at<double>(1,0) = temp_d[1];
    RotationVector.at<double>(2,0) = temp_d[2];
    TranslationVector.at<double>(0,0) = temp_d[3];
    TranslationVector.at<double>(1,0) = temp_d[4];
    TranslationVector.at<double>(2,0) = temp_d[5];
    
    cv::Mat Matrix = cvCreateMat(4,4,CV_64FC1);
    cv::Mat RotationMatrix = cvCreateMat(3,3,CV_64FC1);
    cv::Rodrigues (RotationVector, RotationMatrix);

    for ( int row = 0 ; row < 3 ; row ++ ) 
    {
      for ( int col = 0 ; col < 3 ; col ++ ) 
      {
        Matrix.at<double>(row,col) = RotationMatrix.at<double>(row,col);
      }
    }
   
    for ( int row = 0 ; row < 3 ; row ++ )
    {
      Matrix.at<double>(row,3) = TranslationVector.at<double>(row,0);
    }
    for ( int col = 0 ; col < 3 ; col ++ ) 
    {
      Matrix.at<double>(3,col) = 0.0;
    }
    Matrix.at<double>(3,3) = 1.0;
    myMatrices.push_back(Matrix);
  }
  return myMatrices;
}

//-----------------------------------------------------------------------------
std::vector<cv::Mat> FlipMatrices (const std::vector<cv::Mat> Matrices)
{
  
  std::vector<cv::Mat>  OutMatrices;
  for ( unsigned int i = 0 ; i < Matrices.size() ; i ++ )
  {
    cv::Mat FlipMat = cvCreateMat(4,4,CV_64FC1);
    FlipMat.at<double>(0,0) = Matrices[i].at<double>(0,0);
    FlipMat.at<double>(0,1) = Matrices[i].at<double>(0,1);
    FlipMat.at<double>(0,2) = Matrices[i].at<double>(0,2) * -1;
    FlipMat.at<double>(0,3) = Matrices[i].at<double>(0,3);

    FlipMat.at<double>(1,0) = Matrices[i].at<double>(1,0);
    FlipMat.at<double>(1,1) = Matrices[i].at<double>(1,1);
    FlipMat.at<double>(1,2) = Matrices[i].at<double>(1,2) * -1;
    FlipMat.at<double>(1,3) = Matrices[i].at<double>(1,3);

    FlipMat.at<double>(2,0) = Matrices[i].at<double>(2,0) * -1;
    FlipMat.at<double>(2,1) = Matrices[i].at<double>(2,1) * -1;
    FlipMat.at<double>(2,2) = Matrices[i].at<double>(2,2);
    FlipMat.at<double>(2,3) = Matrices[i].at<double>(2,3) * -1;

    FlipMat.at<double>(3,0) = Matrices[i].at<double>(3,0);
    FlipMat.at<double>(3,1) = Matrices[i].at<double>(3,1);
    FlipMat.at<double>(3,2) = Matrices[i].at<double>(3,2);
    FlipMat.at<double>(3,3) = Matrices[i].at<double>(3,3);

    OutMatrices.push_back(FlipMat);
  }
  return OutMatrices;
}
//-----------------------------------------------------------------------------
std::vector<int> SortMatricesByDistance(const std::vector<cv::Mat>  Matrices)
{
  int NumberOfViews = Matrices.size();

  std::vector<int> used;
  std::vector<int> index;
  for ( int i = 0 ; i < NumberOfViews ; i++ )
  {
    used.push_back(i);
    index.push_back(0);
  }

  int counter = 0;
  int startIndex = 0;
  double distance = 1e-10;

  while ( fabs(distance) > 0 ) 
  {
    cv::Mat t1 = cvCreateMat(3,1,CV_64FC1);
    cv::Mat t2 = cvCreateMat(3,1,CV_64FC1);
    
    for ( int row = 0 ; row < 3 ; row ++ )
    {
      t1.at<double>(row,0) = Matrices[startIndex].at<double>(row,3);
    }
    used [startIndex] = 0 ;
    index [counter] = startIndex;
    counter++;
    distance = 0.0;
    int CurrentIndex=0;
    for ( int i = 0 ; i < NumberOfViews ; i ++ )
    {
      if ( ( startIndex != i ) && ( used[i] != 0 ))
      {
        for ( int row = 0 ; row < 3 ; row ++ )
        {
          t2.at<double>(row,0) = Matrices[i].at<double>(row,3);
        }
        double d = cv::norm(t1-t2);
        if ( d > distance ) 
        {
          distance = d;
          CurrentIndex=i;
        }
      }
    }
    if ( counter < NumberOfViews )
    {
      index[counter] = CurrentIndex;
    }
    startIndex = CurrentIndex;

    
  }
  return index;
}
//-----------------------------------------------------------------------------
std::vector<int> SortMatricesByAngle(const std::vector<cv::Mat>  Matrices)
{
  int NumberOfViews = Matrices.size();

  std::vector<int> used;
  std::vector<int> index;
  for ( int i = 0 ; i < NumberOfViews ; i++ )
  {
    used.push_back(i);
    index.push_back(0);
  }

  int counter = 0;
  int startIndex = 0;
  double distance = 1e-10;

  while ( fabs(distance) > 0.0 ) 
  {
    cv::Mat t1 = cvCreateMat(3,3,CV_64FC1);
    cv::Mat t2 = cvCreateMat(3,3,CV_64FC1);
    
    for ( int row = 0 ; row < 3 ; row ++ )
    {
      for ( int col = 0 ; col < 3 ; col ++ )
      {
        t1.at<double>(row,col) = Matrices[startIndex].at<double>(row,col);
      }
    }
    used [startIndex] = 0 ;
    index [counter] = startIndex;
    counter++;
    distance = 0.0;
    int CurrentIndex=0;
    for ( int i = 0 ; i < NumberOfViews ; i ++ )
    {
      if ( ( startIndex != i ) && ( used[i] != 0 ))
      {
        for ( int row = 0 ; row < 3 ; row ++ )
        {
          for ( int col = 0 ; col < 3 ; col ++ )
          {
            t2.at<double>(row,col) = Matrices[i].at<double>(row,col);
          }
        }
        double d = AngleBetweenMatrices(t1,t2);
        if ( d > distance ) 
        {
          distance = d;
          CurrentIndex=i;
        }
      }
    }
    if ( counter < NumberOfViews )
    {
      index[counter] = CurrentIndex;
    }
    startIndex = CurrentIndex;

    
  }
  return index;
}

//-----------------------------------------------------------------------------
double AngleBetweenMatrices(cv::Mat Mat1 , cv::Mat Mat2)
{
  //turn them into quaternions first
  cv::Mat q1 = DirectionCosineToQuaternion(Mat1);
  cv::Mat q2 = DirectionCosineToQuaternion(Mat2);
  
  return 2 * acos (q1.at<double>(3,0) * q2.at<double>(3,0) 
      + q1.at<double>(0,0) * q2.at<double>(0,0)
      + q1.at<double>(1,0) * q2.at<double>(1,0)
      + q1.at<double>(2,0) * q2.at<double>(2,0));

}

//-----------------------------------------------------------------------------
cv::Mat DirectionCosineToQuaternion(cv::Mat dc_Matrix)
{
  cv::Mat q = cvCreateMat(4,1,CV_64FC1);
  q.at<double>(0,0) = 0.5 * SafeSQRT ( 1 + dc_Matrix.at<double>(0,0) -
      dc_Matrix.at<double>(1,1) - dc_Matrix.at<double>(2,2) ) *
    ModifiedSignum ( dc_Matrix.at<double>(1,2) - dc_Matrix.at<double>(2,1));

  q.at<double>(1,0) = 0.5 * SafeSQRT ( 1 - dc_Matrix.at<double>(0,0) +
      dc_Matrix.at<double>(1,1) - dc_Matrix.at<double>(2,2) ) *
    ModifiedSignum ( dc_Matrix.at<double>(2,0) - dc_Matrix.at<double>(0,2));

  q.at<double>(2,0) = 0.5 * SafeSQRT ( 1 - dc_Matrix.at<double>(0,0) -
      dc_Matrix.at<double>(1,1) + dc_Matrix.at<double>(2,2) ) *
    ModifiedSignum ( dc_Matrix.at<double>(0,1) - dc_Matrix.at<double>(1,0));

  q.at<double>(3,0) = 0.5 * SafeSQRT ( 1 + dc_Matrix.at<double>(0,0) +
      dc_Matrix.at<double>(1,1) + dc_Matrix.at<double>(2,2) );
  
  return q;
}

//-----------------------------------------------------------------------------
double ModifiedSignum(double value)
{
  if ( value < 0.0 ) 
  {
    return -1.0;
  }
  return 1.0;
}
//-----------------------------------------------------------------------------
double SafeSQRT(double value)
{
  if ( value < 0 )
  {
    return 0.0;
  }
  return sqrt(value);
}

} // end namespace
