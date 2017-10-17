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

  std::string uid = niftk::CreateUniqueTempFileName ("tool");
  QString dirName = argv[1] + QString::fromStdString(uid);
  bool isRecording = true;
  niftk::IGIDataSourceI::IGITimeType duration;
  niftk::IGIDataSourceI::IGITimeType timeStamp;
  std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > data;
  mitk::StandaloneDataStorage::Pointer dataStorage = mitk::StandaloneDataStorage::New();
  //give a name, not sure why. Let's make it an individual.
  QString name = "Mark Jackson";

  niftk::IGISingleFileBackend::Pointer backend = niftk::IGISingleFileBackend::New(name,dataStorage.GetPointer());
  MITK_TEST_CONDITION_REQUIRED ( backend.IsNotNull() , "Successfully created IGISingleFileBackend.");

  try
  {
    backend->AddData (dirName, isRecording, duration, timeStamp, data);
    MITK_TEST_CONDITION_REQUIRED ( true , "Adding empty data OK.");
  }
  catch ( std::exception e )
  {
    MITK_TEST_CONDITION_REQUIRED ( false , "Adding empty data called exception:" << e.what());
  }

  mitk::Point4D point;
  mitk::Vector3D vector;
  std::string toolName = "tool";
  std::pair < mitk::Point4D, mitk::Vector3D > pair = std::pair < mitk::Point4D, mitk::Vector3D >(point,vector);

  data.insert ( std::pair < std::string , std::pair<mitk::Point4D, mitk::Vector3D> > ( toolName, pair ) );

  try
  {
    //unless you set expected frames per second prior to calling it, AddData throws an exception.
    backend->SetExpectedFramesPerSecond (1);
    backend->AddData ( dirName, isRecording, duration, timeStamp, data);
    MITK_TEST_CONDITION_REQUIRED ( true , "Adding one data OK.");
  }
  catch ( std::exception e )
  {
    MITK_TEST_CONDITION_REQUIRED ( false , "Adding one data called exception:" << e.what());
  }

  // always end with this!
  MITK_TEST_END();
}
