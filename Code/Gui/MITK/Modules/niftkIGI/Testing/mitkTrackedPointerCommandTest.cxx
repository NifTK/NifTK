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
#include <mitkTrackedPointerCommand.h>
#include <vtkMatrix4x4.h>
#include <vtkLinearTransform.h>
#include <vtkTransform.h>
#include <vtkSmartPointer.h>
#include <mitkDataNode.h>
#include <mitkSurface.h>
#include <mitkCoordinateAxesData.h>

/**
 * \file mitkTrackedPointerCommandTest.cxx.
 * \brief Tests for mitk::TrackedPointerCommand.
 */
int mitkTrackedPointerCommandTest(int /*argc*/, char* /*argv*/[])
{

  double trackingMatrixArray[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 1};
  vtkSmartPointer<vtkMatrix4x4> trackingMatrix = vtkMatrix4x4::New();
  trackingMatrix->DeepCopy(trackingMatrixArray);

  double tipToPointerArray[16] = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 0, 0, 0, 1};
  vtkSmartPointer<vtkMatrix4x4> tipToPointerTransform = vtkMatrix4x4::New();
  tipToPointerTransform->DeepCopy(tipToPointerArray);

  double expectedMatrixArray[16] = {134, 140, 146, 156, 386, 404, 422, 448, 638, 668, 698, 740, 0, 0, 0, 1};
  vtkSmartPointer<vtkMatrix4x4> expectdMatrix = vtkMatrix4x4::New();
  expectdMatrix->DeepCopy(expectedMatrixArray);

  mitk::CoordinateAxesData::Pointer coords = mitk::CoordinateAxesData::New();
  coords->SetVtkMatrix(*trackingMatrix);

  mitk::DataNode::Pointer pointerToWorldNode = mitk::DataNode::New();
  pointerToWorldNode->SetData(coords);

  mitk::Surface::Pointer surface = mitk::Surface::New();

  mitk::DataNode::Pointer surfaceNode = mitk::DataNode::New();
  surfaceNode->SetData(surface);

  mitk::Point3D tip;
  tip[0] = 0;
  tip[1] = 1;
  tip[2] = 2;

  mitk::TrackedPointerCommand::Pointer command = mitk::TrackedPointerCommand::New();
  command->Update(
      tipToPointerTransform,
      pointerToWorldNode,
      surfaceNode,
      tip
      );

  mitk::Point3D expectedTip;
  expectedTip[0] = 588;
  expectedTip[1] = 1696;
  expectedTip[2] = 2804;

  // Check that the point came out in the right place.
  MITK_TEST_CONDITION_REQUIRED(tip[0] == expectedTip[0], ".. Testing x=" << expectedTip[0] << ", but got " << tip[0]);
  MITK_TEST_CONDITION_REQUIRED(tip[1] == expectedTip[1], ".. Testing y=" << expectedTip[1] << ", but got " << tip[1]);
  MITK_TEST_CONDITION_REQUIRED(tip[2] == expectedTip[2], ".. Testing z=" << expectedTip[2] << ", but got " << tip[2]);

  // Check that the matrix was set onto the surface.
  vtkLinearTransform *surfaceGeometryTransform = surface->GetGeometry()->GetVtkTransform();
  vtkSmartPointer<vtkMatrix4x4> surfaceGeometryTransformMatrix = vtkMatrix4x4::New();
  surfaceGeometryTransform->GetMatrix(surfaceGeometryTransformMatrix);

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      double actual = surfaceGeometryTransformMatrix->GetElement(i, j);
      double expected = expectdMatrix->GetElement(i, j);
      MITK_TEST_CONDITION_REQUIRED(actual == expected, ".. Testing expected=" << expected << ", but got " << actual);
    }
  }
  return EXIT_SUCCESS;
}
