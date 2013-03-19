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

#include <iostream>
#include <vtkMatrix4x4.h>
#include <mitkTestingMacros.h>
#include <mitkDataNode.h>
#include <mitkDataStorage.h>
#include <mitkRenderingTestHelper.h>
#include "mitkCoordinateAxesData.h"
#include "mitkCoordinateAxesVtkMapper3D.h"
#include "mitkNifTKCoreObjectFactory.h"

/**
 * \file Test harness for mitk::CoordinateAxesData and mitk::CoordinateAxesVtkMapper3D.
 */
int mitkCoordinateAxesDataRenderingTest(int argc, char * argv[])
{
  // Always start with this, with name of function.
  MITK_TEST_BEGIN("mitkCoordinateAxesDataRenderingTest");

  RegisterNifTKCoreObjectFactory();

  // Create axes node
  mitk::CoordinateAxesData::Pointer axes = mitk::CoordinateAxesData::New();
  mitk::DataNode::Pointer axesNode = mitk::DataNode::New();
  axesNode->SetData(axes);

  // Create a rendering helper - contains window and data storage.
  mitkRenderingTestHelper renderingHelper(640, 480, argc, argv);
  renderingHelper.AddNodeToStorage(axesNode);

  // Set up the view.
  mitk::RenderingManager::GetInstance()->InitializeViews( renderingHelper.GetDataStorage()->ComputeBoundingGeometry3D(renderingHelper.GetDataStorage()->GetAll()) );

  // Get image out
  renderingHelper.Render();
  renderingHelper.SaveAsPNG("/scratch0/NOT_BACKED_UP/clarkson/build/NifTK-SuperBuild-Debug/NifTK-build/output.png");

  MITK_TEST_END();
}

