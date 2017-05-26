/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkAtracsysClientCLP.h>
#include <niftkAtracsysTracker.h>
#include <niftkFileIOUtils.h>
#include <niftkFileHelper.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkException.h>

int main(int argc, char* argv[])
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  try
  {
    if (outputDir.size() == 0)
    {
      commandLine.getOutput()->usage(commandLine);
      return returnStatus;
    }

    mitk::StandaloneDataStorage::Pointer dataStorage = mitk::StandaloneDataStorage::New();
    niftk::AtracsysTracker::Pointer tracker = niftk::AtracsysTracker::New(dataStorage.GetPointer(), toolStorage);

    if (toolStorage.empty())
    {
      for (int c = 0; c < 1000; c++)
      {
        std::vector<mitk::Point3D> points = tracker->GetBallPositions();
        for (int i = 0; i < points.size(); i++)
        {
          MITK_INFO << "niftkAtracsysClient: c=" << c << ", i=" << i << ", p=" 
            << points[i][0] << " "
            << points[i][1] << " "
            << points[i][2] << " "
            << std::endl;
        }
      }
    }
    else
    {
      for (int c = 0; c < 1000; c++)
      {
        std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > markers = tracker->GetTrackingData();
        std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> >::const_iterator iter;
        for (iter = markers.begin(); iter != markers.end(); ++iter)
        {
          MITK_INFO << "niftkAtracsysClient: c=" << c << ", p=" << (*iter).first
            << (*iter).second.first << " "
            << (*iter).second.second << " "
            << std::endl;
        }
      }
    }

    returnStatus = EXIT_SUCCESS;
  }
  catch (mitk::Exception& e)
  {
    MITK_ERROR << "Caught mitk::Exception: " << e.GetDescription() << ", from:" << e.GetFile() << "::" << e.GetLine() << std::endl;
    returnStatus = EXIT_FAILURE + 100;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception: " << e.what() << std::endl;
    returnStatus = EXIT_FAILURE + 101;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:" << std::endl;
    returnStatus = EXIT_FAILURE + 102;
  }
  return returnStatus;
}
