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
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <mitkTestingMacros.h>
#include <mitkVector.h>
#include <mitkIOUtil.h>
#include <mitkDataStorage.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkDataNode.h>

#include <niftkImageOrientation.h>
#include <niftkImageUtils.h>
#include <niftkCoreObjectFactory.h>

/**
 * \brief Test class for niftkImageUtils.
 */
class niftkAsAcquiredOrientationTestClass
{

public:

  //-----------------------------------------------------------------------------
  static void TestAsAcquired(char* argv[])
  {
    MITK_TEST_OUTPUT(<< "Starting TestAsAcquired...");

    // Assume zero arg is program name, first argument is image name, second argument is integer matching enum ImageOrientation
    MITK_TEST_OUTPUT(<< "TestAsAcquired...argv[1]=" << argv[1] << ", argv[2]=" << argv[2] << ", argv[3]=" << argv[3]);

    std::string fileName = argv[1];
    int defaultOrientation = atoi(argv[2]);
    int expectedOrientation = atoi(argv[3]);

    // Need to load image, using MITK utils.
    std::vector<std::string> files;
    files.push_back(fileName);
    mitk::StandaloneDataStorage::Pointer localStorage = mitk::StandaloneDataStorage::New();
    mitk::DataStorage::SetOfObjects::Pointer allImages = mitk::IOUtil::Load(files, *(localStorage.GetPointer()));
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(allImages->size(), 1),".. Testing 1 image loaded.");

    // Get the "As Acquired" orientation.
    const mitk::DataNode::Pointer node = (*allImages)[0];
    bool isImage = niftk::IsImage(node);
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(isImage, true),".. Testing IsImage=true");

    niftk::ImageOrientation orientation = niftk::GetAsAcquiredOrientation(niftk::ImageOrientation(defaultOrientation), dynamic_cast<mitk::Image*>(node->GetData()));
    MITK_TEST_OUTPUT(<< "ImageOrientation default=" << defaultOrientation);
    MITK_TEST_OUTPUT(<< "ImageOrientation output=" << orientation);
    MITK_TEST_OUTPUT(<< "ImageOrientation expected=" << expectedOrientation);
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(orientation, (niftk::ImageOrientation)expectedOrientation),".. Testing expected orientation");

    MITK_TEST_OUTPUT(<< "Finished TestAsAcquired...");
  }
};

/**
 * \brief Basic test harness for a variety of classes, mainly niftkImageUtils
 * that works out the orientation of the XY plane.
 */
int niftkAsAcquiredOrientationTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("niftkAsAcquiredOrientationTest");

  niftkAsAcquiredOrientationTestClass::TestAsAcquired(argv);

  MITK_TEST_END();
}
