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
#include <mitkDataStorage.h>
#include <mitkStandaloneDataStorage.h>
#include <QmitkIGITrackerTool.h>
#include <QmitkIGINiftyLinkDataType.h>
#include <NiftyLinkTrackingDataMessage.h>
#include <QmitkIGITrackerSource.h>

/**
 * \brief This test is simply so we can run through valgrind and check
 * for no leaks.
 */
int QmitkIGINiftyLinkDataSourceMemoryTest(int /*argc*/, char* /*argv*/[])
{

  // Message comes in. Here we create a local pointer, not a smart pointer.
  NiftyLinkTrackingDataMessage* msg = new NiftyLinkTrackingDataMessage();

  // It gets wrapped in a data type. Here we create a local pointer, not a smart pointer.
  QmitkIGINiftyLinkDataType::Pointer dataType = QmitkIGINiftyLinkDataType::New();
  dataType->SetMessage(msg);

  // It gets added to the buffer of the data source.
  mitk::StandaloneDataStorage::Pointer dataStorage = mitk::StandaloneDataStorage::New();
  QmitkIGITrackerSource::Pointer tool = QmitkIGITrackerSource::New(dataStorage, NULL);
  tool->AddData(dataType);

  // When we call delete, the tool should correctly tidy up all memory.
  // When this program itself exits, the smart pointer to tool should delete the tool.
  tool->ClearBuffer();

  return EXIT_SUCCESS;
}
