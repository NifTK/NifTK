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
#include <niftkICPBasedRegistration.h>
#include <mitkDataStorage.h>
#include <mitkIOUtil.h>
#include <mitkCoordinateAxesData.h>
#include <mitkAffineTransformDataNodeProperty.h>
#include <mitkDataStorageUtils.h>
#include <niftkVTKFunctions.h>
#include <vtkTransform.h>
#include <vtkSmartPointer.h>
#include <vtkMinimalStandardRandomSequence.h>

namespace niftk
{
class TestICPBasedRegistration : public ICPBasedRegistration
{
public:
  mitkClassMacro(TestICPBasedRegistration, ICPBasedRegistration);
  itkNewMacro(TestICPBasedRegistration);
  bool SetIndexToWorld(mitk::DataNode::Pointer node , vtkMatrix4x4 * matrix);
  bool CompareMatrices(vtkMatrix4x4 * m1, vtkMatrix4x4 * m2);
  virtual void Initialize(){};
protected:
  virtual ~TestICPBasedRegistration() {};
};

bool TestICPBasedRegistration::SetIndexToWorld(mitk::DataNode::Pointer node , vtkMatrix4x4 * matrix)
{
  mitk::BaseGeometry* geometry = node->GetData()->GetGeometry();
  if (geometry)
  {
    geometry->SetIndexToWorldTransformByVtkMatrix(matrix);
    geometry->Modified();
    return true;
  }
  else
  {
    return false;
  }
}
bool TestICPBasedRegistration::CompareMatrices( vtkMatrix4x4 * m1, vtkMatrix4x4 * m2)
{
  double delta=0.0;
  for ( int i = 0 ; i < 4 ; i ++ ) 
  {
    for ( int j = 0 ; j < 4 ; j ++ )
    {
      delta += fabs(m1->GetElement(i,j) - m2->GetElement(i,j));
    }
  }
  if ( delta > 1e-3 ) 
  {
    std::cerr << "Failed comparing matrices ... " << std::endl <<*m1 << "  ... does not equal" << std::endl << *m2;
    return false;
  }
  else
  {
    return true;
  }
}

} // end namespace

int niftkICPBasedRegistrationTest(int argc, char* argv[])
{
  if (argc != 3)
  {
    std::cerr << "Usage: mitkICPBasedRegistrationTest points.vtp/.mps surface.vtp" << std::endl;
    std::cerr << " argc=" << argc << std::endl;
    for (int i = 0; i < argc; ++i)
    {
      std::cerr << " argv[" << i << "]=" << argv[i] << std::endl;
    }
    return EXIT_FAILURE;
  } 
  
  niftk::TestICPBasedRegistration::Pointer registerer = niftk::TestICPBasedRegistration::New();
  registerer->SetMaximumNumberOfLandmarkPointsToUse(2000);
  registerer->SetMaximumIterations(100);

  // Read Fixed Points
  mitk::DataNode::Pointer fixednode = mitk::DataNode::New();  
  mitk::PointSet::Pointer fixedPoints = mitk::PointSet::New();
  mitk::Surface::Pointer fixedSurface = mitk::Surface::New();

  try
  {
    fixedPoints = mitk::IOUtil::LoadPointSet(argv[1]);
    fixednode->SetData(fixedPoints);
  }
  catch (const mitk::Exception& e)
  {
    // try again, maybe its a surface
    int numberOfPoints = fixedPoints->GetSize();
    if ( numberOfPoints == 0  )
    {
      fixedSurface = mitk::IOUtil::LoadSurface(argv[1]);
      fixednode->SetData(fixedSurface);
    }
  }
  
  // Read Moving Surface
  mitk::DataNode::Pointer movingnode = mitk::DataNode::New();
  mitk::Surface::Pointer movingSurface = mitk::Surface::New();
  movingSurface = mitk::IOUtil::LoadSurface(argv[2]);
  movingnode->SetData(movingSurface);

  // Set up index to world matrices for each node
  vtkSmartPointer<vtkMatrix4x4> fixedMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkSmartPointer<vtkMatrix4x4> movingMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkSmartPointer<vtkMatrix4x4> resultMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  fixedMatrix->Identity();
  movingMatrix->Identity();

  vtkSmartPointer<vtkMinimalStandardRandomSequence> Uni_Rand = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
  Uni_Rand->SetSeed(2);

  // First test, both index to world ID
  if ( registerer->SetIndexToWorld (fixednode, fixedMatrix) && 
      registerer->SetIndexToWorld (movingnode, movingMatrix ) )
  {
    std::cout << "Starting registration test with index2world both identity.";
    registerer->Update(fixednode, movingnode, *resultMatrix);
    mitk::ComposeTransformWithNode(*resultMatrix, movingnode);
    mitk::GetCurrentTransformFromNode(movingnode, *movingMatrix);
    MITK_TEST_CONDITION_REQUIRED(registerer->CompareMatrices(movingMatrix,fixedMatrix), ".. Testing 2 ID");
  }
  else
  {
    return EXIT_FAILURE;
  }

  vtkSmartPointer<vtkTransform> StartTrans = vtkSmartPointer<vtkTransform>::New();

  // Second test, moving non id fixed ID
  niftk::RandomTransform ( StartTrans, 200.0 , 200.0 , 200.0, 50.0 , 50.0, 50.0 , Uni_Rand);

  StartTrans->GetInverse(movingMatrix);
  movingMatrix->Invert();
  fixedMatrix->Identity();
  if ( registerer->SetIndexToWorld (fixednode, fixedMatrix) && 
      registerer->SetIndexToWorld (movingnode, movingMatrix ) )
  {
    std::cout << "Starting registration test with moving index2world non identity.";
    registerer->Update(fixednode, movingnode, *resultMatrix);
    mitk::ComposeTransformWithNode(*resultMatrix, movingnode);
    mitk::GetCurrentTransformFromNode(movingnode, *movingMatrix);
    MITK_TEST_CONDITION_REQUIRED(registerer->CompareMatrices(movingMatrix,fixedMatrix), ".. Testing moving non ID");
  }
  else
  {
    return EXIT_FAILURE;
  }

  // Third test, fixed non id moving ID
  niftk::RandomTransform ( StartTrans, 200.0 , 200.0 , 200.0, 50.0 , 50.0, 50.0 , Uni_Rand);

  StartTrans->GetInverse(fixedMatrix);
  fixedMatrix->Invert();
  movingMatrix->Identity();
  if ( registerer->SetIndexToWorld (fixednode, fixedMatrix) && 
      registerer->SetIndexToWorld (movingnode, movingMatrix ) )
  {
    std::cout << "Starting registration test with fixed index2world non identity.";
    registerer->Update(fixednode, movingnode, *resultMatrix);
    mitk::ComposeTransformWithNode(*resultMatrix, movingnode);
    mitk::GetCurrentTransformFromNode(movingnode, *movingMatrix);
    MITK_TEST_CONDITION_REQUIRED(registerer->CompareMatrices(movingMatrix,fixedMatrix), ".. Testing fixed non ID");
  }
  else
  {
    return EXIT_FAILURE;
  }

  // Forth test, both non id.
  niftk::RandomTransform ( StartTrans, 200.0 , 200.0 , 200.0, 50.0 , 50.0, 50.0 , Uni_Rand);

  StartTrans->GetInverse(fixedMatrix);
  fixedMatrix->Invert();

  niftk::RandomTransform ( StartTrans, 200.0 , 200.0 , 200.0, 50.0 , 50.0, 50.0 , Uni_Rand);

  StartTrans->GetInverse(movingMatrix);
  movingMatrix->Invert();

  if ( registerer->SetIndexToWorld (fixednode, fixedMatrix) && 
      registerer->SetIndexToWorld (movingnode, movingMatrix ) )
  {
    std::cout << "Starting registration test with both index2world non identity.";
    registerer->Update(fixednode, movingnode, *resultMatrix);
    mitk::ComposeTransformWithNode(*resultMatrix, movingnode);
    mitk::GetCurrentTransformFromNode(movingnode, *movingMatrix);
    MITK_TEST_CONDITION_REQUIRED(registerer->CompareMatrices(movingMatrix,fixedMatrix), ".. Testing both non ID");
  }
  else
  {
    return EXIT_FAILURE;
  }

  //still need to test
  //Set rigid, non rigid, 
  //Set number of iterations, 
  //Set maximum number of points
  //
  //Testing of the underlying registration libraries will be done within the
  //relevant libraries. 
  //The testing here only test the plugin functionality, and the correct functioning 
  //of the indexto world transforms.

  return EXIT_SUCCESS;
}
