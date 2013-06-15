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
  virtual void Initialize(){};
protected:
  virtual ~TestSurfaceBasedRegistration() {};
};

} // end namespace

int mitkSurfaceBasedRegistrationTest(int /*argc*/, char* /*argv*/[])
{

  mitk::TestSurfaceBasedRegistration::Pointer registerer = mitk::TestSurfaceBasedRegistration::New();
 // mitk::DataStorage* storage = mitk::DataStorage::New();
  mitk::PointSetReader::Pointer  PointReader = mitk::PointSetReader::New();
  PointReader->SetFileName("/home/thompson/phd/NifIGI-Superbuild-Clean/CMakeExternals/Source/NifTKData/Input/maskedliver_points.mps");
//  PointReader->GenerateData();
  mitk::PointSet::Pointer FixedPoints = mitk::PointSet::New();
  PointReader->Update();
  FixedPoints = PointReader->GetOutput();

  int numberOfPoints = FixedPoints->GetSize();
  std::cout << "There are " << numberOfPoints << "points." << std::endl;
  
  mitk::DataNode::Pointer fixednode = mitk::DataNode::New();
  fixednode->SetData(FixedPoints);

  vtkMatrix4x4 * fixedMatrix = vtkMatrix4x4::New();
  mitk::SurfaceBasedRegistration::Pointer registration = mitk::SurfaceBasedRegistration::New();

  registration->ApplyTransform ( fixednode , fixedMatrix);
  mitk::VtkSurfaceReader::Pointer  SurfaceReader = mitk::VtkSurfaceReader::New();
  SurfaceReader->SetFileName("/home/thompson/phd/NifIGI-Superbuild-Clean/CMakeExternals/Source/NifTKData/Input/liver.vtk");
//  PointReader->GenerateData();
  mitk::Surface::Pointer MovingSurface = mitk::Surface::New();
  SurfaceReader->Update();
  MovingSurface = SurfaceReader->GetOutput();
  
  mitk::DataNode::Pointer movingnode = mitk::DataNode::New();
  movingnode->SetData(MovingSurface);

  vtkMatrix4x4 * movingMatrix = vtkMatrix4x4::New();

  vtkSmartPointer<vtkMinimalStandardRandomSequence> Uni_Rand = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
  Uni_Rand->SetSeed(2);
  vtkSmartPointer<vtkTransform> StartTrans = vtkSmartPointer<vtkTransform>::New();

  RandomTransform ( StartTrans, 10.0 , 10.0 , 10.0, 10.0 , 10.0, 10.0 , Uni_Rand);

  StartTrans->GetInverse(movingMatrix);
  movingMatrix->Invert();

  std::cout << *movingMatrix << std::endl;
  
  registration->ApplyTransform ( movingnode , movingMatrix);
  
  vtkMatrix4x4 * resultMatrix = vtkMatrix4x4::New();
  registration->Update(fixednode, movingnode, resultMatrix);
  std::cerr << "Result with shifted moving" << std::endl;
  std::cerr << *resultMatrix;
  /*registration->ApplyTransform(movingnode);
  mitk::AffineTransform3D::Pointer affineTransform = movingnode->GetData()->GetGeometry()->GetIndexToWorldTransform(); */

  StartTrans->GetInverse(fixedMatrix);
  fixedMatrix->Invert();
  movingMatrix->Identity();
  registration->ApplyTransform ( movingnode , movingMatrix);
  registration->ApplyTransform ( fixednode , fixedMatrix);
  registration->Update(fixednode, movingnode, resultMatrix);
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
  //Test that the the transform respects data nodes world to data transforms.

  return EXIT_SUCCESS;
}
