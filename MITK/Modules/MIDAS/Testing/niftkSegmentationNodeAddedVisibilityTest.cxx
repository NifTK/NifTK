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

#include <cmath>
#include <cstdlib>
#include <iostream>

#include <mitkBaseRenderer.h>
#include <mitkDataNode.h>
#include <mitkDataStorage.h>
#include <mitkGlobalInteraction.h>
#include <mitkImageAccessByItk.h>
#include <mitkIOUtil.h>
#include <mitkRenderingManager.h>
#include <mitkRenderWindow.h>
#include <mitkRenderWindowBase.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkTestingMacros.h>

#include <mitkNifTKCoreObjectFactory.h>
#include <mitkDataNodeAddedVisibilitySetter.h>
#include <niftkToolWorkingDataNameFilter.h>
#include <niftkImageUtils.h>
#include <niftkPolyTool.h>
#include <niftkTool.h>

namespace niftk
{

/**
 * \brief Test class for niftkSegmentationNodeAddedVisibilityTest.
 */
class SegmentationNodeAddedVisibilityTestClass
{

public:

  mitk::DataNode::Pointer m_DataNode;

  //-----------------------------------------------------------------------------
  void Setup()
  {
    MITK_TEST_OUTPUT(<< "Starting Setup...");

    mitk::GlobalInteraction::GetInstance()->Initialize("niftkPaintbrushToolClass");
    m_DataNode = mitk::DataNode::New();

    MITK_TEST_OUTPUT(<< "Finished Setup...");
  }


  //-----------------------------------------------------------------------------
  void TestCreateFilter()
  {
    MITK_TEST_OUTPUT(<< "Starting TestCreateFitler...");

    ToolWorkingDataNameFilter::Pointer filter = ToolWorkingDataNameFilter::New();
    MITK_TEST_CONDITION_REQUIRED(filter.IsNotNull(),".. Testing filter pointer not null.");

    MITK_TEST_OUTPUT(<< "Finished TestCreateFitler...");
  }


  //-----------------------------------------------------------------------------
  void TestFilterPassWithNoPropertiesSet()
  {
    MITK_TEST_OUTPUT(<< "Starting TestFilterPassWithNoPropertiesSet...");

    ToolWorkingDataNameFilter::Pointer filter = ToolWorkingDataNameFilter::New();
    bool result = filter->Pass(m_DataNode);
    MITK_TEST_CONDITION_REQUIRED(result,".. Testing filter passes by default");

    MITK_TEST_OUTPUT(<< "Finished TestFilterPassWithNoPropertiesSet...");
  }


  //-----------------------------------------------------------------------------
  void TestFilterFailWithGivenString(const std::string name)
  {
    MITK_TEST_OUTPUT(<< "Starting TestFilterFailWithGivenString...");

    ToolWorkingDataNameFilter::Pointer filter = ToolWorkingDataNameFilter::New();
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

    mitk::RenderWindow::Pointer renderWindow = mitk::RenderWindow::New(NULL, "niftkPaintbrushToolClass", renderingManager);

    std::vector<const mitk::BaseRenderer*> renderers;
    renderers.push_back(renderWindow->GetRenderer());

    // Create the setter we are testing.
    mitk::DataNodeAddedVisibilitySetter::Pointer setter = mitk::DataNodeAddedVisibilitySetter::New();
    ToolWorkingDataNameFilter::Pointer filter = ToolWorkingDataNameFilter::New();
    setter->AddFilter(filter.GetPointer());
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
    mitk::DataStorage::SetOfObjects::Pointer allImages = mitk::IOUtil::Load(files, *(dataStorage.GetPointer()));
    dataNode = (*allImages)[0];

    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(allImages->size(), 1),".. Testing 1 image loaded.");
    MITK_TEST_CONDITION_REQUIRED(dataNode.IsNotNull(),".. Testing get image pointer.");

    // Check that when the image was loaded it got a global visibility property equal to false.
    bool visibility = true;
    bool foundProperty = false;
    if (doRendererSpecific)
    {
      /// TODO
      /// The const_cast is needed because of the MITK bug 17778. It should be removed after the bug is fixed.
      foundProperty = dataNode->GetBoolProperty("visible", visibility, const_cast<mitk::BaseRenderer*>(renderers[0]));
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

}

/**
 * Basic test harness niftkSegmentationNodeAddedVisibilityTestClass.
 */
int niftkSegmentationNodeAddedVisibilityTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("niftkSegmentationNodeAddedVisibilityTest");

  // We are testing specifically with image ${NIFTK_DATA_DIR}/Input/nv-11x11x11.nii which is 11x11x11.

  niftk::SegmentationNodeAddedVisibilityTestClass *testClass = new niftk::SegmentationNodeAddedVisibilityTestClass();
  testClass->Setup();

  testClass->TestCreateFilter();
  testClass->TestFilterPassWithNoPropertiesSet();
  testClass->TestFilterFailWithGivenString("One of FeedbackContourTool's feedback nodes");
  testClass->TestFilterFailWithGivenString(ContourTool::MIDAS_CONTOUR_TOOL_BACKGROUND_CONTOUR);
  testClass->TestFilterFailWithGivenString(Tool::SEED_POINT_SET_NAME);
  testClass->TestFilterFailWithGivenString(Tool::CURRENT_CONTOURS_NAME);
  testClass->TestFilterFailWithGivenString(Tool::REGION_GROWING_IMAGE_NAME);
  testClass->TestFilterFailWithGivenString(Tool::PRIOR_CONTOURS_NAME);
  testClass->TestFilterFailWithGivenString(Tool::NEXT_CONTOURS_NAME);
  testClass->TestFilterFailWithGivenString(Tool::MORPH_EDITS_EROSIONS_SUBTRACTIONS);
  testClass->TestFilterFailWithGivenString(Tool::MORPH_EDITS_EROSIONS_ADDITIONS);
  testClass->TestFilterFailWithGivenString(Tool::MORPH_EDITS_DILATIONS_SUBTRACTIONS);
  testClass->TestFilterFailWithGivenString(Tool::MORPH_EDITS_DILATIONS_ADDITIONS);
  testClass->TestFilterFailWithGivenString(PolyTool::MIDAS_POLY_TOOL_ANCHOR_POINTS);
  testClass->TestFilterFailWithGivenString(PolyTool::MIDAS_POLY_TOOL_PREVIOUS_CONTOUR);
  testClass->TestFilterFailWithGivenString("Paintbrush_Node");
  testClass->TestVisibilitySetter(argv, false); // global
  testClass->TestVisibilitySetter(argv, true); // renderer specific

  delete testClass;
  MITK_TEST_END();
}

