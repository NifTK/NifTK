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
#include <igtlTimeStamp.h>
#include <iostream>

int main(int argc, char* argv[])
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  // Early exit if main outpt file not specified.
  if (outputFile.size() == 0)
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    std::ofstream opf;
    opf.open(outputFile);
    if (opf.is_open())
    {
      mitk::StandaloneDataStorage::Pointer dataStorage = mitk::StandaloneDataStorage::New();
      niftk::AtracsysTracker::Pointer tracker = niftk::AtracsysTracker::New(dataStorage.GetPointer(), toolStorage);

      unsigned long int counter = 0;
      igtl::TimeStamp::Pointer t = igtl::TimeStamp::New();

      if (toolStorage.empty())
      {
        do
        {
          t->GetTime();
          std::vector<mitk::Point3D> points = tracker->GetBallPositions();

          for (int i = 0; i < points.size(); i++)
          {
            opf << t->GetTimeStampInNanoseconds() << " "
                << i << " " 
                << points[i][0] << " "
                << points[i][1] << " "
                << points[i][2] << " "
                << std::endl;
          }
          if (points.size() > 0)
          {
            counter++;
          }
        } while (counter < numberSamples);
      }
      else
      {
        do
        {
          t->GetTime();
          std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > markers = tracker->GetTrackingData();

          std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> >::const_iterator iter;
          for (iter = markers.begin(); iter != markers.end(); ++iter)
          {
            opf <<  t->GetTimeStampInNanoseconds() << " "
                << (*iter).first << " "
                << (*iter).second.first[0] << " "
                << (*iter).second.first[1] << " "
                << (*iter).second.first[2] << " "
                << (*iter).second.first[3] << " "
                << (*iter).second.second[0] << " "
                << (*iter).second.second[1] << " "
                << (*iter).second.second[2] << " "
                << std::endl;
          }
          if (markers.size() > 0)
          {
            counter++;
          }
        } while (counter < numberSamples);
      }

      returnStatus = EXIT_SUCCESS;

      opf.close();
    }
    else
    {
      std::cerr << "Failed to open file:" << outputFile << std::endl;
      returnStatus = EXIT_SUCCESS + 1;
    }
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
