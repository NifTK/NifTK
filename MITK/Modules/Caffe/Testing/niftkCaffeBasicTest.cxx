/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkCaffeManager.h>

#include <mitkTestingMacros.h>
#include <mitkDataStorage.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkDataNode.h>
#include <mitkIOUtil.h>
#include <caffe/caffe.hpp>

/**
 * \file Test harness for niftk::CaffeManager.
 */
int niftkCaffeBasicTest(int argc, char * argv[])
{
  // Always start with this, with name of function.
  MITK_TEST_BEGIN("niftkCaffeBasicTest");

  if (argc != 6)
  {
    MITK_ERROR << "Usage: niftkCaffeBasicTest network.prototxt weights.caffemodel image.png iterations expectedNumberOfSegmentedPixels";
    return EXIT_FAILURE;
  }

  std::string networkFile = argv[1];
  std::string weightsFile = argv[2];
  std::string inputImageFileName = argv[3];
  int numberIterations = atoi(argv[4]);
  int numberOfExpectedPixels = atoi(argv[5]); // expected number of segmented pixels.

  // Set up network.
  caffe::GlobalInit(&argc, &argv);
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
  caffe::Net<float> caffeNet(networkFile, caffe::TEST);
  caffeNet.CopyTrainedLayersFrom(weightsFile);
/*
  // Load input image, and create node.
  std::vector<std::string> imageFileNames;
  imageFileNames.push_back(inputImageFileName);
  mitk::DataStorage::Pointer dataStorage;
  dataStorage = mitk::StandaloneDataStorage::New();
  mitk::DataStorage::SetOfObjects::Pointer allImages = mitk::IOUtil::Load(imageFileNames, *(dataStorage.GetPointer()));
  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(allImages->size(), 1),".. Testing 1 image loaded.");
  const mitk::DataNode::Pointer inputNode = (*allImages)[0];

  // Do segmentation.
  niftk::CaffeManager::Pointer manager = niftk::CaffeManager::New(networkFile, weightsFile);

  int numberOfActualPixels = 0;
  // Need some way of counting the number of pixels.

  MITK_TEST_CONDITION_REQUIRED(numberOfExpectedPixels == numberOfActualPixels, ".. expect:" << numberOfExpectedPixels << " pixels, but got:" << numberOfActualPixels);
  */
  MITK_TEST_END();
}
