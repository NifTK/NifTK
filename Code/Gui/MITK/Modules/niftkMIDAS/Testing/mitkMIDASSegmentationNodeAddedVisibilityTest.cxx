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
#include <mitkDataStorage.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkIOUtil.h>
#include <mitkDataNode.h>
#include <mitkImageAccessByItk.h>
#include <mitkBaseRenderer.h>
#include <mitkRenderingManager.h>
#include <mitkRenderWindow.h>
#include <mitkRenderWindowBase.h>
#include <mitkGlobalInteraction.h>

#include <mitkNifTKCoreObjectFactory.h>
#include <mitkMIDASImageUtils.h>
#include <mitkMIDASDataNodeNameStringFilter.h>
#include <mitkMIDASNodeAddedVisibilitySetter.h>
#include <mitkMIDASTool.h>
#include <mitkMIDASPolyTool.h>

/**
 * \brief Test class for mitkMIDASSegmentationNodeAddedVisibilityTest.
 */
class mitkMIDASSegmentationNodeAddedVisibilityTestClass
{

public:

  mitk::DataNode::Pointer m_DataNode;

  //-----------------------------------------------------------------------------
  void Setup()
  {
    MITK_TEST_OUTPUT(<< "Starting Setup...");

    // Need to load images, specifically using MIDAS/DRC object factory.
    RegisterNifTKCoreObjectFactory();
    mitk::GlobalInteraction::GetInstance()->Initialize("mitkMIDASPaintbrushToolClass");
    m_DataNode = mitk::DataNode::New();

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


  //-----------------------------------------------------------------------------
  void TestVisibilitySetter(char* argv[], bool doRendererSpecific)
  {
    MITK_TEST_OUTPUT(<< "Starting TestVisibilitySetterGlobal...doRendererSpecific=" << doRendererSpecific);

    mitk::DataStorage::Pointer dataStorage;
    mitk::DataNode::Pointer dataNode;

    // Create local data storage.
    dataStorage = mitk::StandaloneDataStorage::New();
    MITK_TEST_CONDITION_REQUIRED(dataStorage.IsNotNull(),".. Testing created data storage.");

    // Create render window.
    mitk::RenderingManager::Pointer renderingManager = mitk::RenderingManager::GetInstance();
    renderingManager->SetDataStorage(dataStorage);

    mitk::RenderWindow::Pointer renderWindow = mitk::RenderWindow::New(NULL, "mitkMIDASPaintbrushToolClass", renderingManager);

    std::vector< mitk::BaseRenderer* > renderers;
    renderers.push_back(renderWindow->GetRenderer());

    // Create the setter we are testing.
    mitk::MIDASNodeAddedVisibilitySetter::Pointer setter = mitk::MIDASNodeAddedVisibilitySetter::New();
    setter->SetVisibility(false);
    setter->SetDataStorage(dataStorage);
    if (doRendererSpecific)
    {
      setter->SetRenderers(renderers);
    }

    // Load the image.
    std::string fileName = argv[1];
    std::vector<std::string> files;
    files.push_back(fileName);
    mitk::IOUtil::LoadFiles(files, *(dataStorage.GetPointer()));
    mitk::DataStorage::SetOfObjects::ConstPointer allImages = dataStorage->GetAll();
    dataNode = (*allImages)[0];

    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(allImages->size(), 1),".. Testing 1 image loaded.");
    MITK_TEST_CONDITION_REQUIRED(dataNode.IsNotNull(),".. Testing get image pointer.");

    // Check that when the image was loaded it got a global visibility property equal to false.
    bool visibility = true;
    bool foundProperty = false;
    if (doRendererSpecific)
    {
      foundProperty = dataNode->GetBoolProperty("visible", visibility, renderers[0]);
    }
    else
    {
      foundProperty = dataNode->GetBoolProperty("visible", visibility, NULL);
    }
    MITK_TEST_CONDITION_REQUIRED(foundProperty, ".. Testing found property=true");
    MITK_TEST_CONDITION_REQUIRED(!visibility, ".. Testing visibility property=false");

    MITK_TEST_OUTPUT(<< "Finished TestVisibilitySetterGlobal...");
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
  testClass->Setup();

  testClass->TestCreateFilter();
  testClass->TestFilterPassWithNoPropertiesSet();
  testClass->TestFilterFailWithGivenString("FeedbackContourTool");
  testClass->TestFilterFailWithGivenString("MIDASContourTool");
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::SEED_POINT_SET_NAME);
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::CURRENT_CONTOURS_NAME);
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::REGION_GROWING_IMAGE_NAME);
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::PRIOR_CONTOURS_NAME);
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::NEXT_CONTOURS_NAME);
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::MORPH_EDITS_EROSIONS_SUBTRACTIONS);
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::MORPH_EDITS_EROSIONS_ADDITIONS);
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::MORPH_EDITS_DILATIONS_SUBTRACTIONS);
  testClass->TestFilterFailWithGivenString(mitk::MIDASTool::MORPH_EDITS_DILATIONS_ADDITIONS);
  testClass->TestFilterFailWithGivenString(mitk::MIDASPolyTool::MIDAS_POLY_TOOL_ANCHOR_POINTS);
  testClass->TestFilterFailWithGivenString(mitk::MIDASPolyTool::MIDAS_POLY_TOOL_PREVIOUS_CONTOUR);
  testClass->TestFilterFailWithGivenString("Paintbrush_Node");

  testClass->TestVisibilitySetter(argv, false); // global
  testClass->TestVisibilitySetter(argv, true); // renderer specific

  delete testClass;
  MITK_TEST_END();
}

