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
#include <niftkMIDASImageUtils.h>

/**
 * \brief Test class for niftkMIDASImageUtilsTest.
 */
class niftkMIDASImageUtilsTestClass
{

public:



  //-----------------------------------------------------------------------------
  void CopyImage(char* file)
  {
    std::string fileName = file;

    mitk::DataStorage::Pointer dataStorage;
    dataStorage = mitk::StandaloneDataStorage::New();

    // We load the same file 2 times.
    std::vector<std::string> files;
    files.push_back(fileName);
    files.push_back(fileName);

    mitk::DataStorage::SetOfObjects::Pointer allImages = mitk::IOUtil::Load(files, *(dataStorage.GetPointer()));
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(allImages->size(), 2),".. Testing 2 images loaded.");

    const mitk::DataNode::Pointer inputNode = (*allImages)[0];
    mitk::Image::ConstPointer inputImage = dynamic_cast<const mitk::Image*>(inputNode->GetData());

    const mitk::DataNode::Pointer outputNode = (*allImages)[1];
    mitk::Image::Pointer outputImage = dynamic_cast<mitk::Image*>(outputNode->GetData());

    niftk::FillImage(outputImage, 0);
    niftk::CopyIntensityData(inputImage, outputImage);

    bool isEqual = false;
    isEqual = niftk::ImagesHaveEqualIntensities(inputImage, outputImage);
    MITK_TEST_CONDITION_REQUIRED(isEqual, ".. Testing data copied");
  }
};

/**
 * Basic test harness for niftkMIDASImageUtilsTest.
 */
int niftkMIDASImageUtilsTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("niftkMIDASImageUtilsTest");

  niftkMIDASImageUtilsTestClass *testClass = new niftkMIDASImageUtilsTestClass();
  testClass->CopyImage(argv[1]);
  delete testClass;
  MITK_TEST_END();
}

