/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <mitkTestingMacros.h>
#include <mitkStandaloneDataStorage.h>
#include <niftkSpectraTracker.h>
#include <niftkAuroraDomeTracker.h>

/**
 *  \brief Tests connection to NDI trackers.
 *
 * All errors should be thrown as exceptions.
 */
int niftkNDIConnectionTest(int argc , char* argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("niftkNDIConnectionTest")
  MITK_TEST_CONDITION_REQUIRED(argc == 4,"Testing invocation.")

  typedef std::map<std::string, vtkSmartPointer<vtkMatrix4x4> > TrackingData;

  std::string trackerType = argv[1];
  std::string fileNameForConfig = argv[2];
  int numberOfExpectedTools = atoi(argv[3]);

  mitk::StandaloneDataStorage::Pointer dataStorage = mitk::StandaloneDataStorage::New();

  // Uses RAII pattern, so once constructed, it must be valid.
  niftk::NDIBaseTracker::Pointer tracker = NULL;

  if (trackerType == "Polaris")
  {
    niftk::SpectraTracker::Pointer spectraTracker =
        niftk::SpectraTracker::New(
          dataStorage.GetPointer(),        // data storage
          mitk::SerialCommunication::COM5, // port number
          fileNameForConfig                // file name for tools to configure
          );
    tracker = spectraTracker.GetPointer();
  }
  else if (trackerType == "Aurora")
  {
    niftk::AuroraDomeTracker::Pointer auroraTracker =
        niftk::AuroraDomeTracker::New(
          dataStorage.GetPointer(),        // data storage
          mitk::SerialCommunication::COM6, // port number
          fileNameForConfig                // file name for tools to configure
          );
    tracker = auroraTracker.GetPointer();
  }

  // Just test, we can extract right number of tools,
  // and each tool is extracted with name and tracking matrix.
  tracker->StartTracking();
  TrackingData data = tracker->GetTrackingData();
  MITK_TEST_CONDITION_REQUIRED(data.size() == numberOfExpectedTools, ".. Testing number of tracked items=" << numberOfExpectedTools);

  for (int i = 0; i < 1000; i++)
  {
    TrackingData trackingData = tracker->GetTrackingData();
    TrackingData::iterator iter;
    for (iter = trackingData.begin(); iter != trackingData.end(); iter++)
    {
      std::cout << iter->first << " " << *(iter->second) << std::endl;
    }
  }

  // Stops the tracker, before the smart pointer deletes it.
  tracker->StopTracking();

  // always end MITK tests with this!
  MITK_TEST_END();
}

