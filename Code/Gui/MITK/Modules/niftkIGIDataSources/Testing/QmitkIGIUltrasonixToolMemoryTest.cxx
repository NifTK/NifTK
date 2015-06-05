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
#include <NiftyLinkImageMessage.h>
#include <QmitkIGINiftyLinkDataType.h>
#include <QmitkIGIUltrasonixTool.h>
#include <QImage>
#include <QDebug>
#include <QImageReader>
#include <mitkIOUtil.h>
#include <mitkImageReadAccessor.h>

/**
 * \brief This test is simply so we can run through valgrind and check for no leaks.
 * We also test that an image is loaded, and CanHandleData and Update are true.
 */
int QmitkIGIUltrasonixToolMemoryTest(int argc, char* argv[])
{

  if (argc != 2)
  {
    std::cerr << "Usage: QmitkIGIUltrasonixToolMemoryTest imageFileName" << std::endl;
    return EXIT_FAILURE;
  }

  std::cerr << "Loading " << argv[1] << std::endl;

  mitk::Image::Pointer mitkImage = mitk::IOUtil::LoadImage(argv[1]);
  mitk::ImageReadAccessor readAccess(mitkImage);
  const void* vPointer = readAccess.GetData();

  QImage testImage(static_cast<const unsigned char*>(vPointer), mitkImage->GetDimension(0), mitkImage->GetDimension(1), QImage::Format_Indexed8);
  if (testImage.isNull())
  {
    return EXIT_FAILURE;
  }

  NiftyLinkImageMessage::Pointer msg;
  msg = new NiftyLinkImageMessage();
  msg->SetQImage(testImage);

  QmitkIGINiftyLinkDataType::Pointer dataType = QmitkIGINiftyLinkDataType::New();
  dataType->SetMessage(msg.data());

  // It gets added to the buffer of the data source.
  mitk::StandaloneDataStorage::Pointer dataStorage = mitk::StandaloneDataStorage::New();
  QmitkIGIUltrasonixTool::Pointer tool = QmitkIGIUltrasonixTool::New(dataStorage, NULL);
  tool->AddData(dataType);

  bool canHandle = tool->CanHandleData(dataType);
  MITK_TEST_CONDITION_REQUIRED(canHandle == true, ".. Testing if QmitkIGIUltrasonixTool::CanHandleData(QmitkIGINiftyLinkDataType) is true, and value=" << canHandle);

  bool canUpdate = tool->Update(dataType);
  MITK_TEST_CONDITION_REQUIRED(canUpdate == true, ".. Testing if QmitkIGIUltrasonixTool::Update(QmitkIGINiftyLinkDataType) is true, and value=" << canUpdate);

  // When we call delete, the tool should correctly tidy up all memory.
  // When this program itself exits, the smart pointer to tool should delete the tool.
  tool->ClearBuffer();

  return EXIT_SUCCESS;
}
