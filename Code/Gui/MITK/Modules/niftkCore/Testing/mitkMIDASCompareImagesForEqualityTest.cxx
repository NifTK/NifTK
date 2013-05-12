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
#include <mitkImageAccessByItk.h>

#include <mitkNifTKCoreObjectFactory.h>
#include <mitkMIDASImageUtils.h>

/**
 * \brief Test class for mitkMIDASCompareImagesForEqualityTest.
 */
class mitkMIDASCompareImagesForEqualityTestClass
{

public:

  //-----------------------------------------------------------------------------
  static void TestEquality(char* argv[])
  {
    MITK_TEST_OUTPUT(<< "Starting TestEquality...");

    MITK_TEST_OUTPUT(<< "TestAsAcquired...argv[1]=" << argv[1] << ", argv[2]=" << argv[2] << ", argv[3]=" << argv[3]);

    std::string fileName1 = argv[1];
    std::string fileName2 = argv[2];
    int mode = atoi(argv[3]);

    // Need to load images, specifically using MIDAS/DRC object factory.
    RegisterNifTKCoreObjectFactory();

    // Need to load image, using MITK utils.
    std::vector<std::string> files;
    files.push_back(fileName1);
    files.push_back(fileName2);
    mitk::StandaloneDataStorage::Pointer localStorage = mitk::StandaloneDataStorage::New();
    mitk::IOUtil::LoadFiles(files, *(localStorage.GetPointer()));
    mitk::DataStorage::SetOfObjects::ConstPointer allImages = localStorage->GetAll();
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(allImages->size(), 2),".. Testing 2 images loaded.");

    const mitk::DataNode::Pointer node1 = (*allImages)[0];
    const mitk::DataNode::Pointer node2 = (*allImages)[1];

    mitk::Image::ConstPointer image1 = dynamic_cast<const mitk::Image*>(node1->GetData());
    mitk::Image::ConstPointer image2 = dynamic_cast<const mitk::Image*>(node2->GetData());

    bool result = mitk::ImagesHaveEqualIntensities(image1, image2);
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(result, true),".. Testing 2 images equal intensities.");

    if (mode > 0)
    {
      result = mitk::ImagesHaveSameSpatialExtent(image1, image2);
      MITK_TEST_CONDITION_REQUIRED(mitk::Equal(result, true),".. Testing 2 images same spatial extent.");
    }
    MITK_TEST_OUTPUT(<< "Finished TestEquality...");
  }
};

/**
 * \brief Basic test harness to make sure we load DRC Analyze the same as Nifti.
 */
int mitkMIDASCompareImagesForEqualityTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkMIDASCompareImagesForEqualityTest");

  mitkMIDASCompareImagesForEqualityTestClass::TestEquality(argv);

  MITK_TEST_END();
}

