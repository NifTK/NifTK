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

#include <mitkEvaluateIntrinsicParametersOnNumberOfFrames.h>
#include <niftkEvaluateIntrinsicParametersOnNumberOfFramesCLP.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  bool sortByDistance = !DontSortByDistance;
  try
  {
    mitk::Point2D pixelScales;
    pixelScales[0] = pixelScaleFactors[0];
    pixelScales[1] = pixelScaleFactors[1];

    mitk::EvaluateIntrinsicParametersOnNumberOfFrames::Pointer evaluator = mitk::EvaluateIntrinsicParametersOnNumberOfFrames::New();
    evaluator->SetInputDirectory(trackingInputDirectory);
    evaluator->SetInputMatrixDirectory(inputMatrixDirectory);
    evaluator->SetOutputDirectory(outputDirectory);
    evaluator->SetAbsTrackerTimingError(MaxTimingError);
    evaluator->SetFramesToUse(FramesToUse);
    evaluator->SetPixelScaleFactor(pixelScales);
    evaluator->SetSwapVideoChannels(swapVideoChannels);
    evaluator->SetNumberCornersWidth(NumberCornerWidth);
    evaluator->SetNumberCornersHeight(NumberCornerHeight);
    evaluator->InitialiseOutputDirectory();
		
    if ( stepIndicator == 1 ) // only do once to detect the chessboard corners.
    {
      evaluator->InitialiseVideo(); 
    }

    if ( stepIndicator == 2 ) // calling as many as required with different "FramesToUse". Currently number of repetition is fixed in the programe as 20.
    {
      evaluator->RunExperiment();   
    }

    if ( stepIndicator == 3 ) // summary of the experiments.
    {
      evaluator->Report();            
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

  return returnStatus;
}
