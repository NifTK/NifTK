/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkHandeyeCalibrateUsingRegistration.h"
#include "mitkCameraCalibrationFacade.h"
#include <mitkOpenCVMaths.h>
#include <mitkPointSet.h>
#include <mitkIOUtil.h>
#include <mitkFileIOUtils.h>
#include <mitkPointUtils.h>
#include <mitkExceptionMacro.h>
#include <mitkArunLeastSquaresPointRegistrationWrapper.h>
#include <niftkFileHelper.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

namespace mitk {

//-----------------------------------------------------------------------------
HandeyeCalibrateUsingRegistration::HandeyeCalibrateUsingRegistration()
{
}


//-----------------------------------------------------------------------------
HandeyeCalibrateUsingRegistration::~HandeyeCalibrateUsingRegistration()
{

}


//-----------------------------------------------------------------------------
void HandeyeCalibrateUsingRegistration::Calibrate(const std::string& modelInputFile,
  const std::string& modelTrackingDirectory,
  const std::string& cameraPointsDirectory,
  const std::string& handTrackingDirectory,
  const double &distanceThreshold,
  const double &fiducialRegistrationThreshold,
  const std::string& outputMatrixFile
  )
{
  std::vector<cv::Mat> handTrackingMatrices;
  std::vector<cv::Mat> modelTrackingMatrices;
  std::vector<mitk::PointSet::Pointer> cameraPoints;

  if (cameraPointsDirectory.size() == 0)
  {
    mitkThrow() << "No directory specified for reconstucted camera points." << std::endl;
  }

  if (handTrackingDirectory.size() == 0)
  {
    mitkThrow() << "No directory specified for tracking matrices for the 'hand' end (eg. laparoscope)." << std::endl;
  }

  mitk::PointSet::Pointer modelPointSet = mitk::IOUtil::LoadPointSet(modelInputFile);
  if (modelPointSet->GetSize() == 0)
  {
    mitkThrow() << "No model (chessboard) points found in file:" << modelInputFile << std::endl;
  }

  if (modelTrackingDirectory.size() > 0 && niftk::DirectoryExists(modelTrackingDirectory))
  {
    modelTrackingMatrices = mitk::LoadMatricesFromDirectory(modelTrackingDirectory);
  }

  if (cameraPointsDirectory.size() > 0 && niftk::DirectoryExists(cameraPointsDirectory))
  {
    cameraPoints = mitk::LoadPointSetsFromDirectory(cameraPointsDirectory);
  }

  if (cameraPoints.size() == 0)
  {
    mitkThrow() << "No reconstructed point sets were loaded from the directory:" << cameraPointsDirectory << std::endl;
  }

  if (handTrackingDirectory.size() > 0 && niftk::DirectoryExists(handTrackingDirectory))
  {
    handTrackingMatrices = mitk::LoadMatricesFromDirectory(handTrackingDirectory);
  }

  if (handTrackingMatrices.size() == 0)
  {
    mitkThrow() << "No tracking matrices were loaded from the directory:" << handTrackingDirectory << std::endl;
  }

  if (modelTrackingMatrices.size() > 0 && modelTrackingMatrices.size() != cameraPoints.size())
  {
    mitkThrow() << "If model (chessboard) tracking directory is specified, there must be the same number of tracking matrices as the number of sets of reconstucted camera points." << std::endl;
  }

  if (handTrackingMatrices.size() != cameraPoints.size())
  {
    mitkThrow() << "There must be the same number of hand (e.g. laparoscope) tracking matrices as the number of sets of reconstucted camera points." << std::endl;
  }

  vtkSmartPointer<vtkMatrix4x4> calibrationMatrix = vtkMatrix4x4::New();
  calibrationMatrix->Identity();

  this->Calibrate(
    *modelPointSet,
    modelTrackingMatrices,
    handTrackingMatrices,
    cameraPoints,
    distanceThreshold,
    fiducialRegistrationThreshold,
    *calibrationMatrix
    );

  mitk::SaveVtkMatrix4x4ToFile(outputMatrixFile, *calibrationMatrix);
}


//-----------------------------------------------------------------------------
void HandeyeCalibrateUsingRegistration::Calibrate (
  const mitk::PointSet &modelPointSet,
  const std::vector<cv::Mat>& modelTrackingMatrices,
  const std::vector<cv::Mat>& handTrackingMatrices,
  const std::vector<mitk::PointSet::Pointer>& cameraPoints,
  const double& distanceThreshold,
  const double& fiducialRegistrationThreshold,
  vtkMatrix4x4 &outputMatrix
  )
{
  outputMatrix.Identity();

  bool isModelTracking = false;
  if (modelTrackingMatrices.size() > 0 )
  {
    isModelTracking = true;
  }

  if (isModelTracking && modelTrackingMatrices.size() != cameraPoints.size())
  {
    mitkThrow() << "If model (chessboard) tracking matrices are specified, there must be the same number of matrices as the number of sets of reconstucted camera points." << std::endl;
  }

  if (handTrackingMatrices.size() != cameraPoints.size())
  {
    mitkThrow() << "There must be the same number of hand (eg. laparoscope) tracking matrices as the number of sets of reconstucted camera points." << std::endl;
  }

  // Now we basically loop through each camera point set.
  //   We take the model (chessboard) points.
  //     If there are no model tracking matrices, these are assumed to be in tracker space.
  //     If there are tracking matrices, we multiply the model by the corresponding tracker matrix, to convert model points to tracker points.
  //   We then register the model points to the camera points to give us hand (tracker) to eye (camera).
  // Then, we have a whole bunch of registration matrices. Output these.
  // Then we also compute the average using the Frechet norm.

  mitk::PointSet::Pointer modelPointsInTrackerSpace = mitk::PointSet::New();

  mitk::ArunLeastSquaresPointRegistrationWrapper::Pointer pointBasedRegistration = mitk::ArunLeastSquaresPointRegistrationWrapper::New();
  double fiducialRegistrationError = 0;

  cv::Mat trackerToHand = cvCreateMat(4,4,CV_64FC1);
  cv::Mat cameraToTracker = cvCreateMat(4,4,CV_64FC1);
  cv::Mat cameraToHand = cvCreateMat(4,4,CV_64FC1);
  cv::Mat handToCamera = cvCreateMat(4,4,CV_64FC1);
  std::vector< cv::Mat > handEyeMatrices;

  for (unsigned int i = 0; i < cameraPoints.size(); i++)
  {
    vtkSmartPointer<vtkMatrix4x4> trackingTransform = vtkMatrix4x4::New();
    trackingTransform->Identity();

    vtkSmartPointer<vtkMatrix4x4> registrationMatrix = vtkMatrix4x4::New();
    registrationMatrix->Identity();

    if (isModelTracking)
    {
      mitk::CopyToVTK4x4Matrix(modelTrackingMatrices[i], *trackingTransform);
    }

    // This Clears (deletes and repopulates) the modelPointsInTrackerSpace list each time.
    mitk::TransformPointsByVtkMatrix(modelPointSet, *trackingTransform, *modelPointsInTrackerSpace);

    // Gets the first point (origin) of each camera point set... i.e. in camera coordinates.
    // Using an iterator, as the first point is not guaranteed to be labelled as zero.
    mitk::PointSet::DataType* itkPointSet = cameraPoints[i]->GetPointSet();
    mitk::PointSet::PointsContainer* points = itkPointSet->GetPoints();
    mitk::PointSet::PointsIterator pIt;
    mitk::PointSet::PointType origin;
    pIt = points->Begin();
    origin = pIt->Value();

    // Don't use chessboard if origin (first point) is too far away.
    // In this case we know triangulation, and tracking etc. are likely to be unreliable.
    if (fabs(origin[2]) < distanceThreshold)
    {
      pointBasedRegistration->Update(
        modelPointsInTrackerSpace, // fixed points   so this gives us camera-to-tracker
        cameraPoints[i],           // moving points
        *registrationMatrix,
        fiducialRegistrationError
        );

      if (fiducialRegistrationError < fiducialRegistrationThreshold)
      {
        mitk::InvertRigid4x4Matrix(handTrackingMatrices[i], trackerToHand);
        mitk::CopyToOpenCVMatrix(*registrationMatrix, cameraToTracker);
        cameraToHand = trackerToHand * cameraToTracker;
        mitk::InvertRigid4x4Matrix(cameraToHand, handToCamera);
        handEyeMatrices.push_back(handToCamera);

        std::cout << "Hand-eye pair " << i << " registers with FRE=" << fiducialRegistrationError << std::endl;
        for (unsigned int r = 0; r < 4; r++)
        {
          std::cout << handToCamera.at<double>(r, 0) << " " << handToCamera.at<double>(r, 1) << " " << handToCamera.at<double>(r, 2) << " " << handToCamera.at<double>(r, 3) << std::endl;
        }
      }
      else
      {
        std::cout << "Hand-eye pair " << i << " has FRE=" << fiducialRegistrationError << " which is above threshold " << fiducialRegistrationThreshold << " and so is rejected." << std::endl;
      }
    }
    else
    {
      std::cout << "Hand-eye pair " << i << " has z=" << origin[2] << " which is above threshold " << distanceThreshold << " and so is rejected." << std::endl;
    }
  }

  if (handEyeMatrices.size() == 0)
  {
    mitkThrow() << "No suitable registration results were found." << std::endl;
  }

  cv::Mat averageHandeye = cvCreateMat(4,4,CV_64FC1);
  averageHandeye = mitk::AverageMatrices(handEyeMatrices);
  mitk::CopyToVTK4x4Matrix(averageHandeye, outputMatrix);
}

//-----------------------------------------------------------------------------
} // end namespace
