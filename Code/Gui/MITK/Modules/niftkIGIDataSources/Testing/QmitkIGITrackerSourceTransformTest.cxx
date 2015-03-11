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
#include <mitkCoordinateAxesData.h>
#include <QmitkIGINiftyLinkDataType.h>
#include <QmitkIGITrackerSource.h>
#include <NiftyLinkMessageContainer.h>
#include <NiftyLinkTrackingDataMessageHelpers.h>
#include <igtlMath.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

/**
 * \brief Tests if transforms are applied to tracking data.
 */
int QmitkIGITrackerSourceTransformTest(int argc, char* argv[])
{

  if (argc != 1)
  {
    std::cerr << "Usage: QmitkIGITrackerSourceTransformTest" << std::endl;
    return EXIT_FAILURE;
  }

  igtl::Matrix4x4 initialMatrix;
  vtkSmartPointer<vtkMatrix4x4> trackingMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  trackingMatrix->Identity();
  vtkSmartPointer<vtkMatrix4x4> preMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  preMatrix->Identity();
  vtkSmartPointer<vtkMatrix4x4> postMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  postMatrix->Identity();
  vtkSmartPointer<vtkMatrix4x4> tmpMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  tmpMatrix->Identity();

  // Create some test data. Doesn't have to be real rigid body transforms.
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      trackingMatrix->SetElement(i,j, (i+1)*(j+1));
      initialMatrix[i][j] = trackingMatrix->GetElement(i,j);
      initialMatrix[3][j] = trackingMatrix->GetElement(3,j);
      preMatrix->SetElement(i, j, i+j);
      postMatrix->SetElement(i, j, i-j);
    }
  }

  igtl::TimeStamp::Pointer ts = igtl::TimeStamp::New();
  ts->GetTime();

  // The tracking matrix is stored on the message.
  niftk::NiftyLinkMessageContainer::Pointer msg = niftk::CreateTrackingDataMessage(
          QString("TestDevice")
        , QString("TestTool")
        , QString("TestHost")
        , 1234
        , initialMatrix
        , ts
        );


  QmitkIGINiftyLinkDataType::Pointer dataType = QmitkIGINiftyLinkDataType::New();
  dataType->SetMessageContainer(msg);
  dataType->SetTimeStampInNanoSeconds(ts->GetTimeStampInNanoseconds());
  dataType->SetDuration(10000000);

  mitk::StandaloneDataStorage::Pointer dataStorage = mitk::StandaloneDataStorage::New();
  QmitkIGITrackerSource::Pointer tool = QmitkIGITrackerSource::New(dataStorage, NULL);
  tool->AddData(dataType);
  tool->SetPickLatestData(true);

  MITK_TEST_CONDITION_REQUIRED(tool->GetBufferSize()  == 1, ".. Testing if buffer size == 1");

  tool->ProcessData(ts->GetTimeStampInNanoseconds());

  // Check that dataStorage contains a node containing tracking info.
  mitk::DataNode::Pointer node = dataStorage->GetNamedNode("test tracker");
  MITK_TEST_CONDITION_REQUIRED(node.IsNotNull(), ".. Testing if node is not null");

  // Check that it contains a matrix that equals the above matrix.
  mitk::CoordinateAxesData::Pointer coord = dynamic_cast<mitk::CoordinateAxesData*>(node->GetData());
  MITK_TEST_CONDITION_REQUIRED(coord.IsNotNull(), ".. Testing if coord is not null");

  coord->GetVtkMatrix(*tmpMatrix);

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      MITK_TEST_CONDITION_REQUIRED(tmpMatrix->GetElement(i, j) == initialMatrix[i][j], "Checking that i=" << i << ", j=" << j << ", value=" << initialMatrix[i][j] << ", whereas it actuall equals=" << tmpMatrix->GetElement(i, j));
    }
  }

  // Calculate the expected output.
  vtkSmartPointer<vtkMatrix4x4> expectedMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  expectedMatrix->Identity();

  vtkMatrix4x4::Multiply4x4(trackingMatrix, preMatrix, tmpMatrix);
  vtkMatrix4x4::Multiply4x4(postMatrix, tmpMatrix, expectedMatrix);

  for (int i = 0; i < 4; i++)
  {
    std::cerr << "Expected: " << expectedMatrix->GetElement(i, 0) << ", " << expectedMatrix->GetElement(i, 1) << ", " << expectedMatrix->GetElement(i, 2) << ", " << expectedMatrix->GetElement(i, 3) << std::endl;
  }

  // Now stick on the two matrices, and check the result.
  tool->SetPreMultiplyMatrix(*preMatrix);
  tool->SetPostMultiplyMatrix(*postMatrix);

  tool->ProcessData(ts->GetTimeStampInNanoseconds());
  coord->GetVtkMatrix(*tmpMatrix);

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      MITK_TEST_CONDITION_REQUIRED(tmpMatrix->GetElement(i, j) == expectedMatrix->GetElement(i,j), "Checking that i=" << i << ", j=" << j << ", value=" << expectedMatrix->GetElement(i,j) << ", whereas it actually equals=" << tmpMatrix->GetElement(i, j));
    }
  }

  // Now clear matrix, and check result.
  preMatrix->Identity();
  tool->SetPreMultiplyMatrix(*preMatrix);

  postMatrix->Identity();
  tool->SetPostMultiplyMatrix(*postMatrix);

  tool->ProcessData(ts->GetTimeStampInNanoseconds());
  coord->GetVtkMatrix(*tmpMatrix);

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      MITK_TEST_CONDITION_REQUIRED(tmpMatrix->GetElement(i, j) == initialMatrix[i][j], "Checking that i=" << i << ", j=" << j << ", value=" << initialMatrix[i][j] << ", whereas=" << tmpMatrix->GetElement(i, j));
    }
  }

  return EXIT_SUCCESS;
}
