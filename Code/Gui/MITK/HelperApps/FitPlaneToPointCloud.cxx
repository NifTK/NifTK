#include <iostream>
#include <fstream>
#include <typeinfo>
#include <PointClouds/FitPlaneToPointCloudWrapper.h>
#include <FitPlaneToPointCloudCLP.h>


int main(int argc, char* argv[])
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;


  try
  {
    niftk::FitPlaneToPointCloudWrapper::Pointer   fitter = niftk::FitPlaneToPointCloudWrapper::New();
    fitter->FitPlane(pointCloudFileName);

    // dump result to file...
    if (!outputPlaneEquation.empty())
    {
      std::ofstream   logfile(outputPlaneEquation.c_str());
      if (logfile.is_open())
      {
        fitter->OutputParameters(logfile);
      }
      else
      {
        MITK_WARN << "Opening output file failed for some reason!";
      }
      logfile.close();
    }
    // ...and console
    fitter->OutputParameters(std::cerr);

    returnStatus = EXIT_SUCCESS;
  }
  catch (const std::exception& e)
  {
    MITK_ERROR << "Caught '" << typeid(e).name() << "': " << e.what();
  }
  catch (...)
  {
    MITK_ERROR << "Caught exception!";
  }

  return returnStatus;
}
