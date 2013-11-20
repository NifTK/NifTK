#include <iostream>
#include <fstream>
#include <PointClouds/FitPlaneToPointCloudWrapper.h>
#include <FitPlaneToPointCloudCLP.h>


int main(int argc, char* argv[])
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;



  try
  {
    niftk::FitPlaneToPointCloudWrapper::Pointer   fitter = niftk::FitPlaneToPointCloudWrapper::New();
    fitter->FitPlane(pointCloudFileName, outputPlaneEquation);

    returnStatus = EXIT_SUCCESS;
  }
  catch (...)
  {
    MITK_ERROR << "Caught exception!" << std::endl;
  }

  return returnStatus;
}
