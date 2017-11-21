/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <niftkIGISingleFileBackend.h>
#include <niftkFileHelper.h>
#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <mitkStandaloneDataStorage.h>

int niftkIGISingleFileBackendTest ( int argc, char * argv[] )
{
  // always start with this!
  MITK_TEST_BEGIN("niftkIGISingelFileBackendTest");

  std::string uid = niftk::CreateUniqueString ( 6 , time(NULL));
  std::string dirName = std::string(argv[1]) + "tool" +  uid;
  bool isRecording = true;
  niftk::IGIDataSourceI::IGITimeType duration = 10;
  niftk::IGIDataSourceI::IGITimeType timeStamp = 100;
  std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > data;
  mitk::StandaloneDataStorage::Pointer dataStorage = mitk::StandaloneDataStorage::New();
  //give a name, not sure why. Let's make it an individual.
  QString name = "Mark Jackson";

  niftk::IGISingleFileBackend::Pointer backend = niftk::IGISingleFileBackend::New(name,dataStorage.GetPointer());
  MITK_TEST_CONDITION_REQUIRED ( backend.IsNotNull() , "Successfully created IGISingleFileBackend.");

  try
  {
    backend->AddData ( QString::fromStdString(dirName), isRecording, duration, timeStamp, data);
    MITK_TEST_CONDITION_REQUIRED ( true , "Adding empty data OK.");
  }
  catch ( std::exception& e )
  {
    MITK_TEST_CONDITION_REQUIRED ( false , "Adding empty data called exception:" << e.what());
  }

  mitk::Point4D point;
  point[0] = 4;
  point[1] = 44;
  point[2] = 44.4;
  point[3] = 44.44;
  mitk::Vector3D vector;
  vector[0] = -5;
  vector[1] = -55;
  vector[2] = -55.5;
  std::string toolName = "tool";
  std::pair < mitk::Point4D, mitk::Vector3D > pair = std::pair < mitk::Point4D, mitk::Vector3D >(point,vector);

  data.insert ( std::pair < std::string , std::pair<mitk::Point4D, mitk::Vector3D> > ( toolName, pair ) );

//  try
//  {
    //unless you set expected frames per second prior to calling it, AddData throws an exception.
    backend->SetExpectedFramesPerSecond (1);
    backend->AddData ( QString::fromStdString(dirName), isRecording, duration, timeStamp, data);
    MITK_TEST_CONDITION_REQUIRED ( true , "Adding one data OK.");
//  }
//  catch ( std::exception e )
//  {
//    MITK_TEST_CONDITION_REQUIRED ( false , "Adding one data called exception:" << e.what());
//  }

  //must do this to release file for reading.
  backend->StopRecording();

  backend->StartPlayback( QString::fromStdString(dirName), 0, 1 );
  backend->PlaybackData(1,0);

  // always end with this!
  MITK_TEST_END();
}
