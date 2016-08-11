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
#include <mitkDataNode.h>

/**
 * \file Test harness for niftk::CaffeManager.
 */
int niftkCaffeBasicTest(int argc, char * argv[])
{
  // Always start with this, with name of function.
  MITK_TEST_BEGIN("niftkCaffeBasicTest");

  if (argc != 6)
  {
    MITK_ERROR << "Usage: niftkCaffeBasicTest model.prototxt weights.caffemodel image.png iterations expectedNumberOfSegmentedPixels";
    return EXIT_FAILURE;
  }

  std::string modelFile = argv[1];
  std::string weightsFile = argv[2];
  std::string inputImageFileName = argv[3];
  int numberIterations = atoi(argv[4]);
  int numberOfExpectedPixels = atoi(argv[5]); // expected number of segmented pixels.

  // Load input image, and create node.
  mitk::DataNode::ConstPointer inputImage;
  mitk::DataNode::Pointer outputImage = mitk::DataNode::New();

  // Do segmentation.
  niftk::CaffeManager::Pointer manager = niftk::CaffeManager::New(modelFile, weightsFile);

  // To Do: Create some mitk::DataStorage like we do in other unit tests.
  //manager->Segment(dataStorage, inputImage);

  int numberOfActualPixels = 0;
  // Need some way of counting the number of pixels.

  MITK_TEST_CONDITION_REQUIRED(numberOfExpectedPixels == numberOfActualPixels, ".. expect:" << numberOfExpectedPixels << " pixels, but got:" << numberOfActualPixels);
  MITK_TEST_END();
}
