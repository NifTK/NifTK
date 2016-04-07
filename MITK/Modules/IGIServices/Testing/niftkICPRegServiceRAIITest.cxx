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

#include <niftkICPRegServiceRAII.h>
#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <mitkSurface.h>
#include <mitkPointSet.h>
#include <mitkIOUtil.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <stdlib.h>

void BasicICPTest(int argc, char* argv[])
{
  int maxIterations = atoi(argv[3]);
  int maxLandmarks = atoi(argv[4]);

  // Read Fixed Points
  mitk::DataNode::Pointer fixedNode = mitk::DataNode::New();
  mitk::PointSet::Pointer fixedPoints = mitk::PointSet::New();
  fixedPoints = mitk::IOUtil::LoadPointSet(argv[1]);
  fixedNode->SetData(fixedPoints);

  // Read Moving Surface
  mitk::DataNode::Pointer movingNode = mitk::DataNode::New();
  mitk::Surface::Pointer movingSurface = mitk::Surface::New();
  movingSurface = mitk::IOUtil::LoadSurface(argv[2]);
  movingNode->SetData(movingSurface);

  // Just running ICP, and checking it doesn't throw exceptions.
  vtkSmartPointer<vtkMatrix4x4> resultMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  niftk::ICPRegServiceRAII registerer(maxLandmarks, maxIterations);
  registerer.Register(fixedNode, movingNode, *resultMatrix);
}

int niftkICPRegServiceRAIITest ( int argc, char * argv[] )
{
  // always start with this!
  MITK_TEST_BEGIN("niftkICPRegServiceRAIITest");

  if (argc != 5)
  {
    std::cerr << "Usage: niftkICPRegServiceTestRealData points.mps surface.vtp maxIterations maxLandmarks" << std::endl;
    return EXIT_FAILURE;
  }

  std::string envVar = "US_DISABLE_AUTOLOADING";
  const char* input = envVar.c_str();
  const char* result = getenv(input);

  // We can only run these tests if US_DISABLE_AUTOLOADING
  // IS NOT defined. If it is defined, the service wont be loaded.
  if (result == NULL)
  {
    BasicICPTest(argc, argv);
  }
  else
  {
    MITK_INFO << "NOT actually testing anything" << std::endl;
  }

  // always end with this!
  MITK_TEST_END();
}
