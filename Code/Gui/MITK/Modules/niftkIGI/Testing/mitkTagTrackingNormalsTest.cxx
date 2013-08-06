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
#include <mitkTagTrackingRegistrationManager.h>
#include <mitkIOUtil.h>
#include <mitkStereoTagExtractor.h>
#include <mitkPointSet.h>
#include <mitkSurface.h>
#include <mitkCoordinateAxesData.h>
#include <mitkPointUtils.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <Undistortion.h>
/**
 * \file mitkTagTrackingNormalsTest.cxx
 * \brief Tests for a probe containing ARTags.
 */
int mitkTagTrackingNormalsTest(int argc, char* argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkTagTrackingNormalsTest");

  if (argc != 7)
  {
    std::cerr << "Usage = mitkTagTrackingNormalsTest leftImage.nii rightImage.nii leftIntrinsic.txt rightIntrinsic.txt r2lTransform.txt model.vtk" << std::endl;
    for (int i = 0; i < argc; ++i)
    {
      std::cerr << "Arg[" << i << "]=" << argv[i] << std::endl;
    }
    return EXIT_FAILURE;
  }

  std::vector<std::string> files;
  files.push_back(argv[1]);
  files.push_back(argv[2]);
  files.push_back(argv[6]);
  mitk::DataStorage::Pointer dataStorage;
  dataStorage = mitk::StandaloneDataStorage::New();

  // Load images and VTK model, and check we got the right number of data sets.
  mitk::IOUtil::LoadFiles(files, *(dataStorage.GetPointer()));
  mitk::DataStorage::SetOfObjects::ConstPointer allDataItems = dataStorage->GetAll();
  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(allDataItems->size(), 3),".. Testing 2 images and one surface loaded, when actually we have:" << allDataItems->size());

  // Need to load calibration data, so we can correctly extract points and triangulate.
  mitk::DataNode::Pointer leftNode = dataStorage->GetNamedNode("left");
  mitk::DataNode::Pointer rightNode = dataStorage->GetNamedNode("right");
  niftk::Undistortion::LoadIntrinsicCalibration(argv[3], leftNode);
  niftk::Undistortion::LoadIntrinsicCalibration(argv[4], rightNode);
  niftk::Undistortion::LoadStereoRig(argv[5], rightNode);

  // Extract stereo points.
  vtkSmartPointer<vtkMatrix4x4> cameraToWorld = vtkMatrix4x4::New();
  cameraToWorld->Identity();
  mitk::PointSet::Pointer tagCentres = mitk::PointSet::New();
  mitk::PointSet::Pointer tagNormals = mitk::PointSet::New();
  mitk::Image::Pointer leftImage = dynamic_cast<mitk::Image*>(leftNode->GetData());
  mitk::Image::Pointer rightImage = dynamic_cast<mitk::Image*>(rightNode->GetData());

  mitk::StereoTagExtractor::Pointer stereoTagExtractor = mitk::StereoTagExtractor::New();
  stereoTagExtractor->ExtractPoints(
      leftImage,
      rightImage,
      0.005,  // min size
      0.125,  // max size
      20,     // block size
      10,     // offset
      cameraToWorld,
      tagCentres,
      tagNormals
      );

  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(tagCentres->GetSize(), 5),".. Testing 5 points extracted, when actually points=" << tagCentres->GetSize());

  // output point list
  mitk::PointSet::DataType* itkPointSet = tagCentres->GetPointSet(0);
  mitk::PointSet::PointsContainer* points = itkPointSet->GetPoints();
  mitk::PointSet::PointsIterator pIt;
  mitk::PointSet::PointIdentifier pointID;
  mitk::PointSet::PointType point;
  mitk::PointSet::PointType pointArray[3];

  int pointCounter = 0;
  for (pIt = points->Begin(); pIt != points->End(); ++pIt)
  {
    pointID = pIt->Index();
    point = pIt->Value();
    std::cerr << "PointID=" << pointID << ", point=" << point << std::endl;
    pointArray[pointCounter++] = point;
  }

  // FYI.
  std::cerr << "Distance between point[0] and point[1] = " << sqrt(mitk::GetSquaredDistanceBetweenPoints(pointArray[0], pointArray[1])) << std::endl;
  std::cerr << "Distance between point[1] and point[2] = " << sqrt(mitk::GetSquaredDistanceBetweenPoints(pointArray[1], pointArray[2])) << std::endl;

  // Create some input data.
  vtkSmartPointer<vtkMatrix4x4> registrationMatrix = vtkMatrix4x4::New();
  registrationMatrix->Identity();

  mitk::CoordinateAxesData::Pointer transform = mitk::CoordinateAxesData::New();
  mitk::DataNode::Pointer transformNode = mitk::DataNode::New();
  transformNode->SetData(transform);
  transformNode->SetName(mitk::TagTrackingRegistrationManager::TRANSFORM_NODE_ID);
  dataStorage->Add(transformNode);
  mitk::DataNode::Pointer modelNode = dataStorage->GetNamedNode("model.9.tracking");
  MITK_TEST_CONDITION_REQUIRED(modelNode.IsNotNull(), ".. model node is not null");

  // Now we register, using normals.
  mitk::TagTrackingRegistrationManager::Pointer manager = mitk::TagTrackingRegistrationManager::New();
  double fre;
  manager->Update(
      dataStorage,
      tagCentres,
      tagNormals,
      modelNode,
      mitk::TagTrackingRegistrationManager::TRANSFORM_NODE_ID,
      true,
      *registrationMatrix,
      fre
      );

  MITK_TEST_CONDITION_REQUIRED(fre < 2.9,".. Testing FRE is less than 2.9mm, when actually FRE=" << fre);

  MITK_TEST_END();
}


