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
#include <mitkSurfaceBasedRegistration.h>
#include <mitkDataStorage.h>
#include <mitkPointSetReader.h>
#include <mitkVtkSurfaceReader.h>
#include <mitkCoordinateAxesData.h>
#include <mitkAffineTransformDataNodeProperty.h>

#include <vtkFunctions.h>
#include <vtkTransform.h>
#include <vtkSmartPointer.h>
#include <vtkMinimalStandardRandomSequence.h>

namespace mitk
{
class TestSurfaceBasedRegistration : public SurfaceBasedRegistration
{
public:
  mitkClassMacro(TestSurfaceBasedRegistration, SurfaceBasedRegistration);
  itkNewMacro(TestSurfaceBasedRegistration);
  bool SetIndexToWorld(mitk::DataNode::Pointer node , vtkMatrix4x4 * matrix);
  bool CompareMatrices(vtkMatrix4x4 * m1, vtkMatrix4x4 * m2);
  virtual void Initialize(){};
protected:
  virtual ~TestSurfaceBasedRegistration() {};
};

bool TestSurfaceBasedRegistration::SetIndexToWorld(mitk::DataNode::Pointer node , vtkMatrix4x4 * matrix)
{
  mitk::Geometry3D::Pointer geometry = node->GetData()->GetGeometry();
  if (geometry.IsNotNull())
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
bool TestSurfaceBasedRegistration::CompareMatrices( vtkMatrix4x4 * m1, vtkMatrix4x4 * m2)
{
  double delta=0.0;
  for ( int i = 0 ; i < 4 ; i ++ ) 
  {
    for ( int j = 0 ; j < 4 ; j ++ )
    {
      delta += m1->GetElement(i,j) - m2->GetElement(i,j);
    }
  }
  if ( fabs (delta > 1e-3) ) 
  {
    std::cerr << "Failed comparing matrices ... " << std::endl;
    std::cerr << *m1 << "  ... does not equal";
    std::cerr << *m2;
    return false;
  }
  else
  {
    return true;
  }
}

} // end namespace

int mitkSurfaceBasedRegistrationTest(int argc, char* argv[])
{
  if (argc != 3)
  {
    std::cerr << "Usage: mitkSurfaceBasedRegistrationTest points.vtp/.mps surface.vtp" << std::endl;
    std::cerr << " argc=" << argc << std::endl;
    for (int i = 0; i < argc; ++i)
    {
      std::cerr << " argv[" << i << "]=" << argv[i] << std::endl;
    }
    return EXIT_FAILURE;
  } 
  
  mitk::TestSurfaceBasedRegistration::Pointer registerer = mitk::TestSurfaceBasedRegistration::New();
 // mitk::SurfaceBasedRegistration::Pointer registration = mitk::SurfaceBasedRegistration::New();

  //Read Fixed Points
  mitk::PointSetReader::Pointer  PointReader = mitk::PointSetReader::New();
  PointReader->SetFileName(argv[1]);
  mitk::PointSet::Pointer FixedPoints = mitk::PointSet::New();
  PointReader->Update();
  FixedPoints = PointReader->GetOutput();

  int numberOfPoints = FixedPoints->GetSize();
  std::cout << "There are " << numberOfPoints << "points." << std::endl;
  if ( numberOfPoints == 0  )
  {
    std::cerr << "Failed to Read fixed points, hatlting.";
    return EXIT_FAILURE;
  }
  
  mitk::DataNode::Pointer fixednode = mitk::DataNode::New();
  fixednode->SetData(FixedPoints);

  //Read Moving Surface
  mitk::VtkSurfaceReader::Pointer  SurfaceReader = mitk::VtkSurfaceReader::New();
  SurfaceReader->SetFileName(argv[2]);
  mitk::Surface::Pointer MovingSurface = mitk::Surface::New();
  SurfaceReader->Update();
  MovingSurface = SurfaceReader->GetOutput();
  
  mitk::DataNode::Pointer movingnode = mitk::DataNode::New();
  movingnode->SetData(MovingSurface);

  //Set up index to world matrices for each node
  vtkMatrix4x4 * fixedMatrix = vtkMatrix4x4::New();
  vtkMatrix4x4 * movingMatrix = vtkMatrix4x4::New();
  vtkMatrix4x4 * IDMatrix = vtkMatrix4x4::New();
  vtkMatrix4x4 * resultMatrix = vtkMatrix4x4::New();
  fixedMatrix->Identity();
  movingMatrix->Identity();
  IDMatrix->Identity();


  vtkSmartPointer<vtkMinimalStandardRandomSequence> Uni_Rand = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
  Uni_Rand->SetSeed(2);
  //first test, both index to world ID
  if ( registerer->SetIndexToWorld (fixednode, fixedMatrix) && 
      registerer->SetIndexToWorld (movingnode, movingMatrix ) )
  {
    std::cout << "Starting registration test with index2world both identity.";
    registerer->Update(fixednode, movingnode, resultMatrix);
    registerer->ApplyTransform(movingnode, resultMatrix);
    registerer->GetCurrentTransform(movingnode,movingMatrix);
    MITK_TEST_CONDITION_REQUIRED(registerer->CompareMatrices(movingMatrix,fixedMatrix), ".. Testing 2 ID");
  }
  else
  {
    return EXIT_FAILURE;
  }

  vtkSmartPointer<vtkTransform> StartTrans = vtkSmartPointer<vtkTransform>::New();

  RandomTransform ( StartTrans, 10.0 , 10.0 , 10.0, 10.0 , 10.0, 10.0 , Uni_Rand);

  StartTrans->GetInverse(movingMatrix);
  movingMatrix->Invert();

  std::cout << *movingMatrix << std::endl;
  
  
  std::cerr << "Result with shifted moving" << std::endl;
  std::cerr << *resultMatrix;
  /*registration->ApplyTransform(movingnode);
  mitk::AffineTransform3D::Pointer affineTransform = movingnode->GetData()->GetGeometry()->GetIndexToWorldTransform(); */

  StartTrans->GetInverse(fixedMatrix);
  fixedMatrix->Invert();
  movingMatrix->Identity();
  registerer->ApplyTransform ( movingnode , movingMatrix);
  registerer->ApplyTransform ( fixednode , fixedMatrix);
  registerer->Update(fixednode, movingnode, resultMatrix);
  std::cerr << "Result with shifted fixed" << std::endl;
  std::cerr << *resultMatrix;
  //std::cerr << *affineTransform;
 
  //tests
  //load fixed PointSet and fixed surface, and moving surface
  //register for both conditions, 
  //MITK_TEST_CONDITION_REQUIRED(registerGetTrandform() == 1, ".. Testing point to surface");
  //MITK_TEST_CONDITION_REQUIRED(registerGetTrandform() == 1, ".. Testing surface to surface");
  //Set rigid, non rigid, 
  //Set number of iterations, 
  //Set maximum number of points
  //
  //Testing of the underlying registration libraries will be done within the
  //relevant libraries. 
  //The testing here only test the plugin functionality, and the correct functioning 
  //of the indexto world transforms.
  //Index2World Fixed
  //Index2World Moving
  //and transform
  //If both inputs start off with the same index to world and are aligned then the
  //end indextoworld should also be the same
  //Test that the the transform respects data nodes world to data transforms.

  return EXIT_SUCCESS;
}
