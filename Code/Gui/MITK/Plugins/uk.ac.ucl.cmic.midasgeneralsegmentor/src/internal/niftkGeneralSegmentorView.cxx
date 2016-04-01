/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorView.h"

#include <mitkImageStatisticsHolder.h>
#include <mitkIRenderingManager.h>

#include <mitkDataStorageUtils.h>
#include <niftkMIDASContourTool.h>
#include <niftkMIDASTool.h>
#include <niftkGeneralSegmentorGUI.h>

#include "niftkGeneralSegmentorPreferencePage.h"
#include "niftkGeneralSegmentorController.h"

const std::string niftkGeneralSegmentorView::VIEW_ID = "uk.ac.ucl.cmic.midasgeneralsegmentor";

/**************************************************************
 * Start of Constructing/Destructing the View stuff.
 *************************************************************/

//-----------------------------------------------------------------------------
niftkGeneralSegmentorView::niftkGeneralSegmentorView()
: niftkBaseSegmentorView()
, m_GeneralSegmentorGUI(nullptr)
{
}


//-----------------------------------------------------------------------------
niftkGeneralSegmentorView::niftkGeneralSegmentorView(
    const niftkGeneralSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
niftkGeneralSegmentorView::~niftkGeneralSegmentorView()
{
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorController* niftkGeneralSegmentorView::CreateSegmentorController()
{
  m_GeneralSegmentorController = new niftkGeneralSegmentorController(this);
  return m_GeneralSegmentorController;
}


//-----------------------------------------------------------------------------
QString niftkGeneralSegmentorView::GetPreferencesNodeName()
{
  return niftkGeneralSegmentorPreferencePage::PREFERENCES_NODE_NAME;
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::Visible()
{
  niftkBaseSegmentorView::Visible();

  m_GeneralSegmentorController->OnViewGetsVisible();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::Hidden()
{
  niftkBaseSegmentorView::Hidden();

  m_GeneralSegmentorController->OnViewGetsHidden();
}

/**************************************************************
 * End of Constructing/Destructing the View stuff.
 *************************************************************/

/**************************************************************
 * Start of: Some base class functions we have to implement
 *************************************************************/

//-----------------------------------------------------------------------------
std::string niftkGeneralSegmentorView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::SetFocus()
{
  // it seems best not to force the focus, and just leave the
  // focus with whatever the user pressed ... i.e. let Qt handle it.
}

/**************************************************************
 * End of: Some base class functions we have to implement
 *************************************************************/

/**************************************************************
 * Start of: Functions to create reference data (hidden nodes)
 *************************************************************/

//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnNewSegmentationButtonClicked()
{
  assert(m_GeneralSegmentorController);
  niftkBaseSegmentorView::OnNewSegmentationButtonClicked();

  // Create the new segmentation, either using a previously selected one, or create a new volume.
  mitk::DataNode::Pointer newSegmentation = NULL;
  bool isRestarting = false;

  // Make sure we have a reference images... which should always be true at this point.
  mitk::Image* image = this->GetReferenceImageFromToolManager();
  if (image != NULL)
  {
    mitk::ToolManager::Pointer toolManager = this->GetToolManager();
    assert(toolManager);

    mitk::DataNode::Pointer selectedNode = this->GetSelectedNode();

    if (mitk::IsNodeABinaryImage(selectedNode)
        && this->CanStartSegmentationForBinaryNode(selectedNode)
        && !this->IsNodeASegmentationImage(selectedNode)
        )
    {
      newSegmentation =  selectedNode;
      isRestarting = true;
    }
    else
    {
      newSegmentation = this->CreateNewSegmentation(this->GetDefaultSegmentationColor());

      // The above method returns NULL if the user exited the colour selection dialog box.
      if (newSegmentation.IsNull())
      {
        return;
      }
    }

    this->WaitCursorOn();

    // Override the base colour to be orange, and we revert this when OK pressed at the end.
    mitk::Color tmpColor;
    tmpColor[0] = 1.0;
    tmpColor[1] = 0.65;
    tmpColor[2] = 0.0;
    mitk::ColorProperty::Pointer tmpColorProperty = mitk::ColorProperty::New(tmpColor);
    newSegmentation->SetColor(tmpColor);
    newSegmentation->SetProperty("binaryimage.selectedcolor", tmpColorProperty);

    // Set initial properties.
    newSegmentation->SetProperty("layer", mitk::IntProperty::New(90));
    newSegmentation->SetFloatProperty("opacity", 1.0f);
    newSegmentation->SetBoolProperty(niftk::MIDASContourTool::EDITING_PROPERTY_NAME.c_str(), false);

    // Make sure these are up to date, even though we don't use them right now.
    image->GetStatistics()->GetScalarValueMin();
    image->GetStatistics()->GetScalarValueMax();

    // This creates the point set for the seeds.
    mitk::PointSet::Pointer pointSet = mitk::PointSet::New();
    mitk::DataNode::Pointer pointSetNode = mitk::DataNode::New();
    pointSetNode->SetData(pointSet);
    pointSetNode->SetProperty("name", mitk::StringProperty::New(niftk::MIDASTool::SEEDS_NAME));
    pointSetNode->SetFloatProperty("opacity", 1.0f);
    pointSetNode->SetProperty("point line width", mitk::IntProperty::New(1));
    pointSetNode->SetProperty("point 2D size", mitk::IntProperty::New(5));
    pointSetNode->SetBoolProperty("helper object", true);
    pointSetNode->SetBoolProperty("show distant lines", false);
    pointSetNode->SetFloatProperty("Pointset.2D.distance to plane", 0.1);
    pointSetNode->SetBoolProperty("show distances", false);
    pointSetNode->SetProperty("layer", mitk::IntProperty::New(99));
    pointSetNode->SetColor(1.0, 0.0, 0.0);

    // Create all the contours.
    mitk::DataNode::Pointer currentContours = this->CreateContourSet(newSegmentation, 0,1,0, niftk::MIDASTool::CONTOURS_NAME, true, 97);
    mitk::DataNode::Pointer drawContours = this->CreateContourSet(newSegmentation, 0,1,0, niftk::MIDASTool::DRAW_CONTOURS_NAME, true, 98);
    mitk::DataNode::Pointer seeNextNode = this->CreateContourSet(newSegmentation, 0,1,1, niftk::MIDASTool::NEXT_CONTOURS_NAME, false, 95);
    mitk::DataNode::Pointer seePriorNode = this->CreateContourSet(newSegmentation, 0.68,0.85,0.90, niftk::MIDASTool::PRIOR_CONTOURS_NAME, false, 96);

    // Create the region growing image.
    mitk::DataNode::Pointer regionGrowingImageNode = this->CreateHelperImage(image, newSegmentation, 0,0,1, niftk::MIDASTool::REGION_GROWING_NAME, false, 94);

    // Create nodes to store the original segmentation and seeds, so that it can be restored if the Restart button is pressed.
    mitk::DataNode::Pointer initialSegmentationNode = mitk::DataNode::New();
    initialSegmentationNode->SetProperty("name", mitk::StringProperty::New(niftk::MIDASTool::INITIAL_SEGMENTATION_NAME));
    initialSegmentationNode->SetBoolProperty("helper object", true);
    initialSegmentationNode->SetBoolProperty("visible", false);
    initialSegmentationNode->SetProperty("layer", mitk::IntProperty::New(99));
    initialSegmentationNode->SetFloatProperty("opacity", 1.0f);
    initialSegmentationNode->SetColor(tmpColor);
    initialSegmentationNode->SetProperty("binaryimage.selectedcolor", tmpColorProperty);

    mitk::DataNode::Pointer initialSeedsNode = mitk::DataNode::New();
    initialSeedsNode->SetProperty("name", mitk::StringProperty::New(niftk::MIDASTool::INITIAL_SEEDS_NAME));
    initialSeedsNode->SetBoolProperty("helper object", true);
    initialSeedsNode->SetBoolProperty("visible", false);
    initialSeedsNode->SetBoolProperty("show distant lines", false);
    initialSeedsNode->SetFloatProperty("Pointset.2D.distance to plane", 0.1);
    initialSeedsNode->SetBoolProperty("show distances", false);
    initialSeedsNode->SetProperty("layer", mitk::IntProperty::New(99));
    initialSeedsNode->SetColor(1.0, 0.0, 0.0);

    /// TODO
    /// We should not refer to mitk::RenderingManager::GetInstance() because the DnD display uses its
    /// own rendering manager, not this one, like the MITK display.
    mitk::IRenderingManager* renderingManager = 0;
    mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();
    if (renderWindowPart)
    {
      renderingManager = renderWindowPart->GetRenderingManager();
    }
    if (renderingManager)
    {
      // Make sure these points and contours are not rendered in 3D, as there can be many of them if you "propagate",
      // and furthermore, there seem to be several seg faults rendering contour code in 3D. Haven't investigated yet.
      QList<vtkRenderWindow*> renderWindows = renderingManager->GetAllRegisteredVtkRenderWindows();
      for (QList<vtkRenderWindow*>::const_iterator iter = renderWindows.begin(); iter != renderWindows.end(); ++iter)
      {
        if ( mitk::BaseRenderer::GetInstance((*iter))->GetMapperID() == mitk::BaseRenderer::Standard3D )
        {
          pointSetNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
          seePriorNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
          seeNextNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
          currentContours->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
          drawContours->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
          initialSegmentationNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
          initialSeedsNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
        }
      }
    }

    // Adding to data storage, where the ordering affects the layering.
    this->GetDataStorage()->Add(seePriorNode, newSegmentation);
    this->GetDataStorage()->Add(seeNextNode, newSegmentation);
    this->GetDataStorage()->Add(regionGrowingImageNode, newSegmentation);
    this->GetDataStorage()->Add(currentContours, newSegmentation);
    this->GetDataStorage()->Add(drawContours, newSegmentation);
    this->GetDataStorage()->Add(pointSetNode, newSegmentation);
    this->GetDataStorage()->Add(initialSegmentationNode, newSegmentation);
    this->GetDataStorage()->Add(initialSeedsNode, newSegmentation);

    // Set working data. See header file, as the order here is critical, and should match the documented order.
    mitk::ToolManager::DataVectorType workingData(9);
    workingData[niftk::MIDASTool::SEGMENTATION] = newSegmentation;
    workingData[niftk::MIDASTool::SEEDS] = pointSetNode;
    workingData[niftk::MIDASTool::CONTOURS] = currentContours;
    workingData[niftk::MIDASTool::DRAW_CONTOURS] = drawContours;
    workingData[niftk::MIDASTool::PRIOR_CONTOURS] = seePriorNode;
    workingData[niftk::MIDASTool::NEXT_CONTOURS] = seeNextNode;
    workingData[niftk::MIDASTool::REGION_GROWING] = regionGrowingImageNode;
    workingData[niftk::MIDASTool::INITIAL_SEGMENTATION] = initialSegmentationNode;
    workingData[niftk::MIDASTool::INITIAL_SEEDS] = initialSeedsNode;
    toolManager->SetWorkingData(workingData);

    if (isRestarting)
    {
      this->InitialiseSeedsForWholeVolume();
      this->UpdateCurrentSliceContours();
    }

    this->StoreInitialSegmentation();

    // Setup GUI.
    m_GeneralSegmentorGUI->SetAllWidgetsEnabled(true);
    m_GeneralSegmentorGUI->SetThresholdingWidgetsEnabled(false);
    m_GeneralSegmentorGUI->SetThresholdingCheckBoxEnabled(true);
    m_GeneralSegmentorGUI->SetThresholdingCheckBoxChecked(false);

    this->FocusOnCurrentWindow();
    this->OnFocusChanged();
    this->RequestRenderWindowUpdate();

    this->WaitCursorOff();

  } // end if we have a reference image

  m_GeneralSegmentorController->m_IsRestarting = isRestarting;

  // Finally, select the new segmentation node.
  this->SetCurrentSelection(newSegmentation);
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer niftkGeneralSegmentorView::CreateContourSet(mitk::DataNode::Pointer segmentationNode, float r, float g, float b, std::string name, bool visible, int layer)
{
  assert(m_GeneralSegmentorController);
  return m_GeneralSegmentorController->CreateContourSet(segmentationNode, r, g, b, name, visible, layer);
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer niftkGeneralSegmentorView::CreateHelperImage(mitk::Image::Pointer referenceImage, mitk::DataNode::Pointer segmentationNode, float r, float g, float b, std::string name, bool visible, int layer)
{
  assert(m_GeneralSegmentorController);
  return m_GeneralSegmentorController->CreateHelperImage(referenceImage, segmentationNode, r, g, b, name, visible, layer);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::StoreInitialSegmentation()
{
  assert(m_GeneralSegmentorController);
  m_GeneralSegmentorController->StoreInitialSegmentation();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::UpdateCurrentSliceContours(bool updateRendering)
{
  assert(m_GeneralSegmentorController);
  return m_GeneralSegmentorController->UpdateCurrentSliceContours(updateRendering);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::InitialiseSeedsForWholeVolume()
{
  assert(m_GeneralSegmentorController);
  m_GeneralSegmentorController->InitialiseSeedsForWholeVolume();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::onVisibilityChanged(const mitk::DataNode* node)
{
  assert(m_GeneralSegmentorController);
  m_GeneralSegmentorController->OnNodeVisibilityChanged(node);
}


/**************************************************************
 * End of: Functions to create reference data (hidden nodes)
 *************************************************************/


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnFocusChanged()
{
  QmitkBaseView::OnFocusChanged();

  assert(m_GeneralSegmentorController);
  m_GeneralSegmentorController->OnFocusChanged();
}




//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::NodeChanged(const mitk::DataNode* node)
{
  assert(m_GeneralSegmentorController);
  m_GeneralSegmentorController->OnNodeChanged(node);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::NodeRemoved(const mitk::DataNode* node)
{
  assert(m_GeneralSegmentorController);
  m_GeneralSegmentorController->OnNodeRemoved(node);
}
