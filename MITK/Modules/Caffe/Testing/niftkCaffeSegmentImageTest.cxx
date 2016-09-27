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
#include <caffe/caffe.hpp>
#include <boost/tokenizer.hpp>
#include <highgui.h>

/**
 * \file Test harness for niftk::CaffeManager.
 */
int niftkCaffeSegmentImageTest(int argc, char * argv[])
{
  // Always start with this, with name of function.
  MITK_TEST_BEGIN("niftkCaffeSegmentImageTest");

  if (argc != 12)
  {
    MITK_ERROR << "Usage: niftkCaffeSegmentImageTest network.prototxt weights.caffemodel outputLayerName inputSizeX inputSizeY outputSizeX outputSizeY meanRed,meanGreen,meanBlue inputImage.png expectedOutputImage.png actualOutputImage.png";
    return EXIT_FAILURE;
  }

  std::string networkFile = argv[1];
  std::string weightsFile = argv[2];
  std::string outputLayerName = argv[3];
  int inputSizeX = atoi(argv[4]);
  int inputSizeY = atoi(argv[5]);
  int outputSizeX = atoi(argv[6]);
  int outputSizeY = atoi(argv[7]);
  std::string offsetString = argv[8];
  std::string inputImageFileName = argv[9];
  std::string expectedOutputImageFileName = argv[10];
  std::string actualOutputImageFileName = argv[11]; // written to, for debugging.

  float offsets[3];
  boost::char_separator<char> sep(",");
  boost::tokenizer<boost::char_separator<char> > tokens(offsetString, sep);
  int counter = 0;
  for (const auto& t : tokens) {
    offsets[counter] = atof(t.c_str());
    counter++;
  }
  MITK_TEST_CONDITION_REQUIRED(counter == 3, "... Testing 3 comma separated values for mean RGB during training.");

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

  int dummyArgc = 3; // caffe command line parser doesn't like negative arguments.
  caffe::GlobalInit(&dummyArgc, &argv);

  mitk::Vector3D offsetRGB;
  offsetRGB[0] = offsets[0];
  offsetRGB[1] = offsets[1];
  offsetRGB[2] = offsets[2];

  mitk::Point2I inputSize;
  inputSize[0] = inputSizeX;
  inputSize[1] = inputSizeY;

  mitk::Point2I outputSize;
  outputSize[0] = outputSizeX;
  outputSize[1] = outputSizeY;

  niftk::CaffeFCNSegmentor::Pointer manager =
    niftk::CaffeFCNSegmentor::New(networkFile,
                                  weightsFile,
                                  offsetRGB,
                                  inputSize,
                                  outputLayerName,
                                  outputSize
                                 );

  manager->Segment(inputImage, outputImage);

  // For debugging:
  mitk::IOUtil::Save(outputImage, actualOutputImageFileName);

  // Check exact match.
  MITK_TEST_CONDITION_REQUIRED(niftk::ImagesHaveEqualIntensities(outputImage, expectedOutputImage), "... Checking outputImage and expectedOutputImage have equal intensities.");

  MITK_TEST_END();
}
