/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
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

#include "mitkMIDASEnums.h"
#include "mitkMIDASImageUtils.h"
#include "mitkNifTKCoreObjectFactory.h"

/**
 * \brief Test class for mitkMIDASImageUtils.
 */
class mitkMIDASAsAcquiredOrientationTestClass
{

public:

  //-----------------------------------------------------------------------------
  static void TestAsAcquired(char* argv[])
  {
    MITK_TEST_OUTPUT(<< "Starting TestAsAcquired...");

    // Assume zero arg is program name, first argument is image name, second argument is integer matching enum MIDASView
    MITK_TEST_OUTPUT(<< "TestAsAcquired...argv[1]=" << argv[1] << ", argv[2]=" << argv[2] << ", argv[3]=" << argv[3]);

    std::string fileName = argv[1];
    int defaultView = atoi(argv[2]);
    int expectedView = atoi(argv[3]);

    // Need to load images, specifically using MIDAS/DRC object factory.
    RegisterNifTKCoreObjectFactory();

    // Need to load image, using MITK utils.
    std::vector<std::string> files;
    files.push_back(fileName);
    mitk::StandaloneDataStorage::Pointer localStorage = mitk::StandaloneDataStorage::New();
    mitk::IOUtil::LoadFiles(files, *(localStorage.GetPointer()));
    mitk::DataStorage::SetOfObjects::ConstPointer allImages = localStorage->GetAll();
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(allImages->size(), 1),".. Testing 1 image loaded.");

    // Get the "As Acquired" view.
    const mitk::DataNode::Pointer image = (*allImages)[0];
    MIDASView view = mitk::GetAsAcquiredView(MIDASView(defaultView), image);
    MITK_TEST_OUTPUT(<< "MIDASView default=" << defaultView);
    MITK_TEST_OUTPUT(<< "MIDASView output=" << view);
    MITK_TEST_OUTPUT(<< "MIDASView expected=" << expectedView);
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(view, (MIDASView)expectedView),".. Testing expected view");

    MITK_TEST_OUTPUT(<< "Finished TestAsAcquired...");
  }
};

/**
 * \brief Basic test harness for a variety of classes, mainly mitkMIDASImageUtils
 * that works out the orientation of the XY plane.
 */
int mitkMIDASAsAcquiredOrientationTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkMIDASAsAcquiredOrientationTest");

  mitkMIDASAsAcquiredOrientationTestClass::TestAsAcquired(argv);

  MITK_TEST_END();
}

