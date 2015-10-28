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

#include <mitkHandeyeCalibrateFromDirectory.h>
#include <niftkHandeyeCalibrationFromDirectoryCLP.h>
#include <niftkArunLeastSquaresPointRegistration.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <mitkPointSet.h>
#include <mitkIOUtil.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  bool sortByDistance = !dontSortByDistance;
  try
  {
    mitk::Point2D pixelScales;
    pixelScales[0] = pixelScaleFactors[0];
    pixelScales[1] = pixelScaleFactors[1];

    mitk::HandeyeCalibrateFromDirectory::Pointer calibrator = mitk::HandeyeCalibrateFromDirectory::New();
    calibrator->SetInputDirectory(trackingInputDirectory);
    calibrator->SetOutputDirectory(outputDirectory);
    calibrator->SetTrackerIndex(trackerIndex);
    calibrator->SetAbsTrackerTimingError(maxTimingError);
    calibrator->SetFramesToUse(framesToUse);
    calibrator->SetFramesToUseFactor(framesToUseFactor);
    calibrator->SetStickToFramesToUse(stickToFramesToUse);
    calibrator->SetSortByDistance(sortByDistance);
    calibrator->SetFlipTracking(flipTracking);
    calibrator->SetFlipExtrinsic(flipExtrinsic);
    calibrator->SetSortByAngle(false);
    calibrator->SetPixelScaleFactor(pixelScales);
    calibrator->SetSwapVideoChannels(swapVideoChannels);
    calibrator->SetNumberCornersWidth(numberCornerWidth);
    calibrator->SetNumberCornersHeight(numberCornerHeight);
    calibrator->SetSquareSizeInMillimetres(squareSizeInMM);
    calibrator->SetRandomise(randomise);

    // If the user specified a chessboard (in tracker coordinates), we do direct registration method.
    // This is only really for testing purposes. ToDo: Put code somewhere more sensible.
    vtkSmartPointer<vtkMatrix4x4> chessboardToTracker = vtkSmartPointer<vtkMatrix4x4>::New();
    chessboardToTracker->Identity();
    if (chessboardPoints.size() > 0)
    {
      mitk::PointSet::Pointer chessboardPointsInTrackerCoordinates = mitk::IOUtil::LoadPointSet(chessboardPoints);
      mitk::PointSet::Pointer chessboardPointsInModelCoordinates = mitk::PointSet::New();

      // Assume for now, we are doing 4 corners.
      mitk::Point3D p;
      p[0] = 0;
      p[1] = 0;
      p[2] = 0;
      chessboardPointsInModelCoordinates->InsertPoint(p);

      p[0] = (numberCornerWidth-1)*squareSizeInMM;
      chessboardPointsInModelCoordinates->InsertPoint(p);

      p[1] = (numberCornerHeight-1)*squareSizeInMM;
      chessboardPointsInModelCoordinates->InsertPoint(p);

      p[0] = 0;
      chessboardPointsInModelCoordinates->InsertPoint(p);

      // Register model points to tracker coordinates
      niftk::PointBasedRegistrationUsingSVD(chessboardPointsInTrackerCoordinates, chessboardPointsInModelCoordinates, *chessboardToTracker);

      // If successful, pass to calibrator.
      calibrator->SetChessBoardToTracker(chessboardToTracker);
    }

    calibrator->InitialiseOutputDirectory();
    calibrator->InitialiseTracking();

    if ( existingIntrinsicsDirectory != "" )
    {
      MITK_INFO << "Attempting to use existing intrinsic calibration from " << existingIntrinsicsDirectory;
      calibrator->LoadExistingIntrinsicCalibrations(existingIntrinsicsDirectory);
    }

    if ( existingRightToLeftDirectory != "" )
    {
      MITK_INFO << "Attempting to use existing right-to-left calibration from " << existingRightToLeftDirectory;
      calibrator->LoadExistingRightToLeft(existingRightToLeftDirectory);
    }

    calibrator->InitialiseVideo();

    returnStatus = EXIT_SUCCESS;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception:" << e.what();
    returnStatus = -1;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:";
    returnStatus = -2;
  }

  return returnStatus;
}
