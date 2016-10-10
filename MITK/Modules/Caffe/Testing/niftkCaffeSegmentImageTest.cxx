/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkCaffeFCNSegmentor.h>
#include <niftkImageUtils.h>

#include <mitkTestingMacros.h>
#include <mitkDataStorage.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkDataNode.h>
#include <mitkIOUtil.h>
#define GLOG_NO_ABBREVIATED_SEVERITIES
#pragma push_macro("STRICT")
#undef STRICT
#include <caffe/caffe.hpp>
#pragma pop_macro("STRICT")
#include <chrono>
#include <ctime>

/**
 * \file Test harness for niftk::CaffeManager.
 */
int niftkCaffeSegmentImageTest(int argc, char * argv[])
{
  // Always start with this, with name of function.
  MITK_TEST_BEGIN("niftkCaffeSegmentImageTest");

  if (argc != 8)
  {
    MITK_ERROR << "Usage: niftkCaffeSegmentImageTest network.prototxt weights.caffemodel inputLayerName outputBlobName inputImage.png expectedOutputImage.png actualOutputImage.png";
    return EXIT_FAILURE;
  }

  std::string networkFile = argv[1];
  std::string weightsFile = argv[2];
  std::string inputLayerName = argv[3];
  std::string outputBlobName = argv[4];
  std::string inputImageFileName = argv[5];
  std::string expectedOutputImageFileName = argv[6];
  std::string actualOutputImageFileName = argv[7]; // written to, for debugging.

  // Load input image, and create node.
  std::vector<std::string> imageFileNames;
  imageFileNames.push_back(inputImageFileName);
  imageFileNames.push_back(expectedOutputImageFileName);
  mitk::DataStorage::Pointer dataStorage;
  dataStorage = mitk::StandaloneDataStorage::New();
  mitk::DataStorage::SetOfObjects::Pointer allImages = mitk::IOUtil::Load(imageFileNames, *(dataStorage.GetPointer()));

  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(allImages->size(), 2),"... Testing 2 images loaded.");
  const mitk::DataNode::Pointer inputNode = (*allImages)[0];
  const mitk::DataNode::Pointer expectedOutputNode = (*allImages)[1];

  const mitk::Image::Pointer inputImage = dynamic_cast<mitk::Image*>(inputNode->GetData());
  MITK_TEST_CONDITION_REQUIRED(inputImage.IsNotNull(), "... Testing input image is in fact an image.");

  const mitk::Image::Pointer expectedOutputImage = dynamic_cast<mitk::Image*>(expectedOutputNode->GetData());
  MITK_TEST_CONDITION_REQUIRED(expectedOutputImage.IsNotNull(), "... Testing expected output image is in fact an image.");
  MITK_TEST_CONDITION_REQUIRED(expectedOutputImage->GetNumberOfChannels() == 1, "... Testing I have 1 channel output, as it should be a mask [0|255]");

  const mitk::Image::Pointer outputImage = expectedOutputImage->Clone();

  caffe::GlobalInit(&argc, &argv);

  niftk::CaffeFCNSegmentor::Pointer manager =
    niftk::CaffeFCNSegmentor::New(networkFile,
                                  weightsFile,
                                  inputLayerName,
                                  outputBlobName,
                                  -1 // no gpu
                                 );

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  manager->Segment(inputImage, outputImage);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end-start;
  std::cout << "elapsed time: " << elapsed.count() << "s\n";

  start = std::chrono::system_clock::now();

  manager->Segment(inputImage, outputImage);

  end = std::chrono::system_clock::now();
  elapsed = end-start;
  std::cout << "elapsed time: " << elapsed.count() << "s\n";

  // For debugging:
  mitk::IOUtil::Save(outputImage, actualOutputImageFileName);

  // Check exact match.
  MITK_TEST_CONDITION_REQUIRED(niftk::ImagesHaveEqualIntensities(outputImage, expectedOutputImage), "... Checking outputImage and expectedOutputImage have equal intensities.");

  MITK_TEST_END();
}
