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
#include <mitkHandeyeCalibrate.h>
#include <mitkCameraCalibrationFacade.h>
#include <niftkHandeyeCalibrationCLP.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;
  std::vector<double> ReprojectionError;

  if ( trackingInputDirectory.length() == 0 ||
    ! ( ( extrinsicInputDirectory.length() == 0 ) != ( extrinsicInputFile.length() == 0 )) )
  {
    if ( fullHandeyeInputDirectory.length() == 0 )
    {
      std::cout << trackingInputDirectory.length() << " " << extrinsicInputDirectory.length() << " " << extrinsicInputFile.length() << std::endl;
      commandLine.getOutput()->usage(commandLine);
      return returnStatus;
    }
    else
    {
      try 
      {
        mitk::GenerateFullHandeyeMatrices(fullHandeyeInputDirectory);
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
  }
 
  bool FlipExtrin = FlipExtrinsics;
  bool SortByDistance = ! ( DontSortByDistance || SortByAngle );

  try
  {
    mitk::HandeyeCalibrate::Pointer calibrationObject = mitk::HandeyeCalibrate::New();
    calibrationObject->SetFlipTracking(FlipTracking);
    calibrationObject->SetFlipExtrinsic(FlipExtrin);
    calibrationObject->SetSortByDistance(SortByDistance);
    calibrationObject->SetSortByAngle(SortByAngle);
    calibrationObject->SetNumberCornersWidth(NumberCornerWidth);
    calibrationObject->SetNumberCornersHeight(NumberCornerHeight);
    calibrationObject->SetSquareSizeInMillimetres(squareSizeInmm);

    if ( extrinsicInputDirectory.length() == 0 )
    {
      ReprojectionError = calibrationObject->Calibrate(trackingInputDirectory,
        extrinsicInputFile);
    }
    else
    {
      ReprojectionError = calibrationObject->Calibrate(trackingInputDirectory,
        extrinsicInputDirectory);
    }
    if ( fullHandeyeInputDirectory.length() != 0 )
    {
      mitk::GenerateFullHandeyeMatrices(fullHandeyeInputDirectory);
    } 
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

  std::cout << "Reprojection error=" << ReprojectionError[0] << ", return status = " << returnStatus << std::endl;
  return returnStatus;
}
