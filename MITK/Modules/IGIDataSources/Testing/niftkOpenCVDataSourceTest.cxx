/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <mitkTestingMacros.h>
#include <mitkStandaloneDataStorage.h>
#include <niftkIGIDataSourceFactoryServiceRAII.h>
#include <niftkIGIDataSourceI.h>
#include <QApplication>

/**
 * \file mitkTrackedImage.cxx
 * \brief Tests for the OpenCV video data source.
 *
 * Note that we are NOT linking to OpenCV. Its all done via Services.
 */
int niftkOpenCVDataSourceTest(int argc, char* argv[])
{

  QApplication app(argc, argv);
  Q_UNUSED(app);

  try
  {
    mitk::StandaloneDataStorage::Pointer dataStorage = mitk::StandaloneDataStorage::New();
    niftk::IGIDataSourceFactoryServiceRAII factory("OpenCVVideoDataSourceFactory");

    niftk::IGIDataSourceProperties props;
    niftk::IGIDataSourceI* source = factory.CreateService(dataStorage.GetPointer(), props);
    source->SetRecordingLocation("/tmp/matt");
    source->StartRecording();

  } catch (mitk::Exception& e)
  {
    // Expected to fail if no webcam is present,
    // so, this test harness is only really for manual testing.
    MITK_WARN << "niftkOpenCVDataSourceTest failed due to:" << e.what();
  }

  return EXIT_SUCCESS;
}
