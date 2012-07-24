/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-24 08:23:13 +0100 (Tue, 24 Jul 2012) $
 Revision          : $Revision: 9382 $
 Last modified by  : $Author: mjc $

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
#include <mitkDataStorage.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkIOUtil.h>
#include <mitkDataNode.h>
#include <mitkImageAccessByItk.h>

#include "mitkNifTKCoreObjectFactory.h"
#include "mitkMIDASImageUtils.h"
#include "mitkMIDASDataNodeNameStringFilter.h"
#include "mitkMIDASTool.h"
#include "mitkMIDASPolyTool.h"

/**
 * \brief Test class for mitkMIDASSegmentationNodeAddedVisibilityTest.
 */
class mitkMIDASSegmentationNodeAddedVisibilityTestClass
{

public:

  mitk::DataStorage::Pointer m_DataStorage;
  mitk::DataNode::Pointer m_DataNode;

  //-----------------------------------------------------------------------------
  void Setup(char* argv[])
  {
    MITK_TEST_OUTPUT(<< "Starting Setup...");

    std::string fileName = argv[1];

    // Need to load images, specifically using MIDAS/DRC object factory.
    RegisterNifTKCoreObjectFactory();

    // Load them into a local data storage.
    m_DataStorage = mitk::StandaloneDataStorage::New();

    // Load the image.
    std::vector<std::string> files;
    files.push_back(fileName);

    mitk::IOUtil::LoadFiles(files, *(m_DataStorage.GetPointer()));
    mitk::DataStorage::SetOfObjects::ConstPointer allImages = m_DataStorage->GetAll();
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(allImages->size(), 1),".. Testing 1 image loaded.");

    m_DataNode = (*allImages)[0];
    MITK_TEST_CONDITION_REQUIRED(m_DataNode.IsNotNull(),".. Testing get image pointer.");

    MITK_TEST_OUTPUT(<< "Finished Setup...");
  }


  //-----------------------------------------------------------------------------
  void TestCreateFilter()
  {
    MITK_TEST_OUTPUT(<< "Starting TestCreateFitler...");

    mitk::MIDASDataNodeNameStringFilter::Pointer filter = mitk::MIDASDataNodeNameStringFilter::New();
    MITK_TEST_CONDITION_REQUIRED(filter.IsNotNull(),".. Testing filter pointer not null.");

    MITK_TEST_OUTPUT(<< "Finished TestCreateFitler...");
  }


  //-----------------------------------------------------------------------------
  void TestFilterPassWithNoPropertiesSet()
  {
    MITK_TEST_OUTPUT(<< "Starting TestFilterPassWithNoPropertiesSet...");

    mitk::MIDASDataNodeNameStringFilter::Pointer filter = mitk::MIDASDataNodeNameStringFilter::New();
    bool result = filter->Pass(m_DataNode);
    MITK_TEST_CONDITION_REQUIRED(result,".. Testing filter passes by default");

    MITK_TEST_OUTPUT(<< "Finished TestFilterPassWithNoPropertiesSet...");
  }


  //-----------------------------------------------------------------------------
  void TestFilterFailWithGivenString(const std::string name)
  {
    MITK_TEST_OUTPUT(<< "Starting TestFilterFailWithGivenString...");

    mitk::MIDASDataNodeNameStringFilter::Pointer filter = mitk::MIDASDataNodeNameStringFilter::New();
    m_DataNode->SetName(name);
    bool result = filter->Pass(m_DataNode);
    MITK_TEST_CONDITION_REQUIRED(!result,".. Testing filter fails with name=" << name);

    MITK_TEST_OUTPUT(<< "Finished TestFilterFailWithGivenString...");
  }
};

/**
 * Basic test harness mitkMIDASSegmentationNodeAddedVisibilityTestClass.
 */
int mitkMIDASSegmentationNodeAddedVisibilityTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkMIDASSegmentationNodeAddedVisibilityTest");

  // We are testing specifically with image ${NIFTK_DATA_DIR}/Input/nv-11x11x11.nii which is 11x11x11.

  mitkMIDASSegmentationNodeAddedVisibilityTestClass *testClass = new mitkMIDASSegmentationNodeAddedVisibilityTestClass();
  testClass->Setup(argv);
  testClass->TestCreateFilter();
  testClass->TestFilterPassWithNoPropertiesSet();
  testClass->TestFilterFailWithGivenString("FeedbackContourTool");
  testClass->TestFilterFailWithGivenString("MIDASContourTool");
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::SEED_POINT_SET_NAME);
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::CURRENT_CONTOURS_NAME);
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::REGION_GROWING_IMAGE_NAME);
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::PRIOR_CONTOURS_NAME);
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::NEXT_CONTOURS_NAME);
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::MORPH_EDITS_SUBTRACTIONS);
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::MORPH_EDITS_ADDITIONS);
  testClass->TestFilterFailWithGivenString(mitk::MIDASPolyTool::MIDAS_POLY_TOOL_ANCHOR_POINTS);
  testClass->TestFilterFailWithGivenString(mitk::MIDASPolyTool::MIDAS_POLY_TOOL_PREVIOUS_CONTOUR);
  testClass->TestFilterFailWithGivenString("Paintbrush_Node");

  delete testClass;
  MITK_TEST_END();
}

