#include <iostream>
#include <fstream>
#include <PointClouds/FitPlaneToPointCloudWrapper.h>


int main(int argc, char* argv[])
{
  if (argc < 1)
  {
    std::cerr << "specify an mitk mps pointset file!" << std::endl;
    return 1;
  }


  try
  {
    niftk::FitPlaneToPointCloudWrapper::Pointer   fitter = niftk::FitPlaneToPointCloudWrapper::New();

  }
  catch (...)
  {
    std::cerr << "Caught exception!" << std::endl;
  }


}
