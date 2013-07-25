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
#include <mitkCoordinateAxesData.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <Undistortion.h>
/**
 * \file mitkTagTrackingAndRegistrationTest.cxx
 * \brief Tests for mitk::TrackedImageCommand.
 */
int mitkTagTrackingAndRegistrationTest(int argc, char* argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkTagTrackingAndRegistrationTest");

  if (argc != 7)
  {
    std::cerr << "Usage = mitkTagTrackingAndRegistrationTest leftImage rightImage leftIntrinsic rightIntrinsic r2lTransform model" << std::endl;
    for (int i = 0; i < argc; ++i)
    {
      std::cerr << "Arg[" << i << "]=" << argv[i] << std::endl;
    }
    return EXIT_FAILURE;
  }

  std::vector<std::string> files;
  files.push_back(argv[1]);
  files.push_back(argv[2]);
  mitk::DataStorage::Pointer dataStorage;
  dataStorage = mitk::StandaloneDataStorage::New();


  // Load images, and check we got two images.
  mitk::IOUtil::LoadFiles(files, *(dataStorage.GetPointer()));
  mitk::DataStorage::SetOfObjects::ConstPointer allImages = dataStorage->GetAll();
  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(allImages->size(), 2),".. Testing 2 image loaded.");

  // Need to load calibration data, so we can correctly extract points and triangulate.
  mitk::DataNode::Pointer leftNode = dataStorage->GetNamedNode("NVIDIA-SDI-stream-0-undistorted");
  mitk::DataNode::Pointer rightNode = dataStorage->GetNamedNode("NVIDIA-SD-stream-1-undistorted");
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

  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(tagCentres->GetSize(), 32),".. Testing 32 points extracted, when actually points=" << tagCentres->GetSize());

  // output point list
  mitk::PointSet::DataType* itkPointSet = tagCentres->GetPointSet(0);
  mitk::PointSet::PointsContainer* points = itkPointSet->GetPoints();
  mitk::PointSet::PointsIterator pIt;
  mitk::PointSet::PointIdentifier pointID;
  mitk::PointSet::PointType point;

  for (pIt = points->Begin(); pIt != points->End(); ++pIt)
  {
    pointID = pIt->Index();
    point = pIt->Value();
    std::cerr << "PointID=" << pointID << ", point=" << point << std::endl;
  }

  // Also load the model (point set)
  mitk::DataNode::Pointer modelNode = mitk::DataNode::New();
  mitk::PointSet::Pointer modelPoints = mitk::IOUtil::LoadPointSet(argv[6]);
  modelNode->SetData(modelPoints);
  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(modelPoints->GetSize(), 18),".. Testing 18 model points loaded, when actually modelPoints=" << modelPoints->GetSize());

  // Now we register.
  vtkSmartPointer<vtkMatrix4x4> registrationMatrix = vtkMatrix4x4::New();
  registrationMatrix->Identity();
  mitk::CoordinateAxesData::Pointer transform = mitk::CoordinateAxesData::New();
  mitk::DataNode::Pointer transformNode = mitk::DataNode::New();
  transformNode->SetData(transform);
  transformNode->SetName(mitk::TagTrackingRegistrationManager::TRANSFORM_NODE_ID);

  dataStorage->Add(transformNode);

  // Testing the point based registration, no normals used.
  mitk::TagTrackingRegistrationManager::Pointer manager = mitk::TagTrackingRegistrationManager::New();
  double fre = manager->Update(
      dataStorage,
      tagCentres,
      tagNormals,
      modelNode,
      mitk::TagTrackingRegistrationManager::TRANSFORM_NODE_ID,
      false,
      *registrationMatrix
      );

  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(fre, 0.1),".. Testing FRE is just about equal to 0.1, when actually FRE=" << fre);

  MITK_TEST_END();
}


