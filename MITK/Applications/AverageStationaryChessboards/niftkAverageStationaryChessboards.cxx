/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <mitkExceptionMacro.h>
#include <mitkPointSet.h>
#include <mitkIOUtil.h>

#include <mitkOpenCVFileIOUtils.h>
#include <mitkFileIOUtils.h>
#include <mitkCameraCalibrationFacade.h>
#include <niftkFileHelper.h>
#include <niftkAverageStationaryChessboardsCLP.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if (    leftCameraInputDirectory.length() == 0
       || rightCameraInputDirectory.length() == 0
       || intrinsicLeft.length() == 0
       || intrinsicRight.length() == 0
       || rightToLeftExtrinsics.length() == 0
       || outputPoints.length() == 0
       )
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    mitk::Point2D pixelScales;
    pixelScales[0] = pixelScaleFactors[0];
    pixelScales[1] = pixelScaleFactors[1];

    std::vector<std::string> leftFiles = niftk::GetFilesInDirectory(leftCameraInputDirectory);
    if (leftFiles.size() == 0)
    {
      std::ostringstream errorMessage;
      errorMessage << "No files in directory:" << leftCameraInputDirectory << std::endl;
      mitkThrow() << errorMessage.str();
    }

    std::vector<std::string> rightFiles = niftk::GetFilesInDirectory(rightCameraInputDirectory);
    if (rightFiles.size() == 0)
    {
      std::ostringstream errorMessage;
      errorMessage << "No files in directory:" << rightCameraInputDirectory << std::endl;
      mitkThrow() << errorMessage.str();
    }

    if (leftFiles.size() != rightFiles.size())
    {
      std::ostringstream errorMessage;
      errorMessage << "Different number of files in left:" << leftCameraInputDirectory << ", and right:" <<  rightCameraInputDirectory << std::endl;
      mitkThrow() << errorMessage.str();
    }

    std::sort(leftFiles.begin(), leftFiles.end());
    std::sort(rightFiles.begin(), rightFiles.end());

    std::vector<std::string> successfulLeftFiles;
    std::vector<std::string> successfulRightFiles;

    for (int i = 0; i < leftFiles.size(); i++)
    {
      mitk::CheckAndAppendPairOfFileNames(leftFiles[i],
                                          rightFiles[i],
                                          xCorners,
                                          yCorners,
                                          1, // not needed as we triangulate
                                          pixelScales,
                                          successfulLeftFiles,
                                          successfulRightFiles);
    }

    // Sanity check
    if (successfulLeftFiles.size() == 0)
    {
      mitkThrow() << "No successful chessboards in left camera" << std::endl;
    }
    if (successfulRightFiles.size() == 0)
    {
      mitkThrow() << "No successful chessboards in right camera" << std::endl;
    }
    if (successfulLeftFiles.size() != successfulRightFiles.size())
    {
      mitkThrow() << "The left and right channel had a different number of images with successfully matched corners.";
    }

    std::vector<IplImage*> imagesLeft;
    std::vector<std::string> fileNamesLeft;
    std::vector<IplImage*> imagesRight;
    std::vector<std::string> fileNamesRight;
    std::vector<IplImage*> successfullImagesLeft;
    std::vector<std::string> successfullFileNamesLeft;
    std::vector<IplImage*> successfullImagesRight;
    std::vector<std::string> successfullFileNamesRight;

    CvMat *imagePointsLeft = NULL;
    CvMat *objectPointsLeft = NULL;
    CvMat *pointCountsLeft = NULL;

    CvMat *imagePointsRight = NULL;
    CvMat *objectPointsRight = NULL;
    CvMat *pointCountsRight = NULL;

    std::cout << "AverageStationaryChessboards: Loading left" << std::endl;
    mitk::LoadImages(successfulLeftFiles, imagesLeft, fileNamesLeft);

    std::cout << "AverageStationaryChessboards: Loading right" << std::endl;
    mitk::LoadImages(successfulRightFiles, imagesRight, fileNamesRight);

    std::vector<IplImage*> allImages;
    int width = 0;
    int height = 0;

    allImages.insert(allImages.begin(), imagesLeft.begin(), imagesLeft.end());
    allImages.insert(allImages.begin(), imagesRight.begin(), imagesRight.end());

    mitk::CheckConstImageSize(allImages, width, height);

    mitk::ExtractChessBoardPoints(imagesLeft,
                                  fileNamesLeft,
                                  xCorners,
                                  yCorners,
                                  false,
                                  1,
                                  pixelScales,
                                  successfullImagesLeft,
                                  successfullFileNamesLeft,
                                  imagePointsLeft,
                                  objectPointsLeft,
                                  pointCountsLeft);

    mitk::ExtractChessBoardPoints(imagesRight,
                                  fileNamesRight,
                                  xCorners,
                                  yCorners,
                                  false,
                                  1,
                                  pixelScales,
                                  successfullImagesRight,
                                  successfullFileNamesRight,
                                  imagePointsRight,
                                  objectPointsRight,
                                  pointCountsRight);

    // Sanity check
    assert(imagePointsLeft->rows == imagePointsRight->rows);
    assert(imagePointsLeft->cols == imagePointsRight->cols);
    assert(objectPointsLeft->rows == objectPointsRight->rows);
    assert(objectPointsLeft->cols == objectPointsRight->cols);
    assert(pointCountsLeft->rows == pointCountsRight->rows);
    assert(pointCountsLeft->cols == pointCountsRight->cols);
    assert(imagePointsLeft->rows == xCorners*yCorners*fileNamesLeft.size());
    assert(imagePointsRight->rows == xCorners*yCorners*fileNamesRight.size());
    assert(objectPointsLeft->rows == xCorners*yCorners*fileNamesLeft.size());
    assert(objectPointsRight->rows == xCorners*yCorners*fileNamesRight.size());

    // Load calibration data
    cv::Mat leftIntrinsic = cvCreateMat (3,3,CV_64FC1);
    cv::Mat leftDistortion = cvCreateMat (1,4,CV_64FC1);              // not used (yet)
    cv::Mat rightIntrinsic = cvCreateMat (3,3,CV_64FC1);
    cv::Mat rightDistortion = cvCreateMat (1,4,CV_64FC1);             // not used (yet)
    cv::Mat rightToLeftRotationMatrix = cvCreateMat (3,3,CV_64FC1);
    cv::Mat rightToLeftTranslationVector = cvCreateMat (1,3,CV_64FC1);

    // Load matrices. These throw exceptions if things fail.
    mitk::LoadCameraIntrinsicsFromPlainText(intrinsicLeft, &leftIntrinsic, &leftDistortion);
    mitk::LoadCameraIntrinsicsFromPlainText(intrinsicRight, &rightIntrinsic, &rightDistortion);
    mitk::LoadStereoTransformsFromPlainText(rightToLeftExtrinsics, &rightToLeftRotationMatrix, &rightToLeftTranslationVector);

    unsigned int numberOfImages = fileNamesLeft.size();
    unsigned int expectedNumberOfPointsPerImage = xCorners*yCorners;
    unsigned int expectedNumberOfPoints = expectedNumberOfPointsPerImage*numberOfImages;

    // Allocate 2D points
    std::vector< std::pair<cv::Point2d, cv::Point2d> > pointPairs;
    pointPairs.resize(expectedNumberOfPointsPerImage);
    for (long int i = 0; i < expectedNumberOfPointsPerImage; i++)
    {
      pointPairs[i].first.x = 0;
      pointPairs[i].first.y = 0;
      pointPairs[i].second.x = 0;
      pointPairs[i].second.y = 0;
    }

    // Average Points in 2D
    unsigned int imageIndex = 0;

    for (unsigned int i = 0; i < expectedNumberOfPoints; i++)
    {
      imageIndex = i % expectedNumberOfPointsPerImage;

      pointPairs[imageIndex].first.x += CV_MAT_ELEM(*imagePointsLeft, double, i, 0);
      pointPairs[imageIndex].first.y += CV_MAT_ELEM(*imagePointsLeft, double, i, 1);
      pointPairs[imageIndex].second.x += CV_MAT_ELEM(*imagePointsRight, double, i, 0);
      pointPairs[imageIndex].second.y += CV_MAT_ELEM(*imagePointsRight, double, i, 1);
    }
    for (unsigned int i = 0; i < expectedNumberOfPointsPerImage; i++)
    {
      pointPairs[i].first.x /= static_cast<double>(numberOfImages);
      pointPairs[i].first.y /= static_cast<double>(numberOfImages);
      pointPairs[i].second.x /= static_cast<double>(numberOfImages);
      pointPairs[i].second.y /= static_cast<double>(numberOfImages);
    }

    // Triangulate
    std::vector <std::pair < cv::Point3d, double > > pointsIn3D = mitk::TriangulatePointPairsUsingGeometry(
        pointPairs,
        leftIntrinsic,
        rightIntrinsic,
        rightToLeftRotationMatrix,
        rightToLeftTranslationVector,
        // choose an arbitrary threshold that is unlikely to overflow.
        std::numeric_limits<int>::max());

    // Write out
    if (niftk::FilenameHasPrefixAndExtension(outputPoints, "", "mps"))
    {
      mitk::PointSet::Pointer pointSet = mitk::PointSet::New();
      for (unsigned int i = 0; i < pointsIn3D.size(); i++)
      {
        mitk::Point3D point;
        point[0] = pointsIn3D[i].first.x;
        point[1] = pointsIn3D[i].first.y;
        point[2] = pointsIn3D[i].first.z;
        pointSet->InsertPoint(i, point);
      }
      mitk::IOUtil::Save(pointSet, outputPoints);
    }
    else
    {
      std::ostringstream errorMessage;
      errorMessage << "Not implemented output to formats other than .mpsm (yet)." << std::endl;
      mitkThrow() << errorMessage.str();
    }

    // Done
    returnStatus = EXIT_SUCCESS;
  }
  catch (std::exception& e)
  {
    std::cerr << "Caught std::exception:" << e.what();
    returnStatus = -1;
  }
  catch (...)
  {
    std::cerr << "Caught unknown exception:";
    returnStatus = -2;
  }
  return returnStatus;
}
