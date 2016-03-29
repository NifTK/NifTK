/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorView.h"

#include <QButtonGroup>
#include <QGridLayout>
#include <QMessageBox>

#include <itkCommand.h>
#include <itkContinuousIndex.h>
#include <itkImageFileWriter.h>

#include <mitkColorProperty.h>
#include <mitkContourModelSet.h>
#include <mitkDataNodeObject.h>
#include <mitkDataStorageUtils.h>
#include <mitkFocusManager.h>
#include <mitkGlobalInteraction.h>
#include <mitkImageAccessByItk.h>
#include <mitkImageStatisticsHolder.h>
#include <mitkIRenderingManager.h>
#include <mitkITKImageImport.h>
#include <mitkOperationEvent.h>
#include <mitkPlaneGeometry.h>
#include <mitkPointSet.h>
#include <mitkPointUtils.h>
#include <mitkProperties.h>
#include <mitkRenderingManager.h>
#include <mitkSegmentationObjectFactory.h>
#include <mitkSegTool2D.h>
#include <mitkSlicedGeometry3D.h>
#include <mitkStringProperty.h>
#include <mitkSurface.h>
#include <mitkTool.h>
#include <mitkUndoController.h>
#include <mitkVtkResliceInterpolationProperty.h>

#include <QmitkRenderWindow.h>

#include <niftkGeneralSegmentorPipeline.h>
#include <niftkGeneralSegmentorPipelineCache.h>
#include <niftkGeneralSegmentorUtils.h>
#include <niftkMIDASDrawTool.h>
#include <niftkMIDASImageUtils.h>
#include <niftkMIDASOrientationUtils.h>
#include <niftkMIDASPolyTool.h>
#include <niftkMIDASPosnTool.h>
#include <niftkMIDASSeedTool.h>
#include <niftkMIDASTool.h>

#include <niftkGeneralSegmentorCommands.h>

#include "niftkGeneralSegmentorController.h"
#include <niftkGeneralSegmentorGUI.h>

/*
#include <sys/time.h>
double timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p)
{
  return (((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) -
           ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec))/1000000000.0;
}
*/

const std::string niftkGeneralSegmentorView::VIEW_ID = "uk.ac.ucl.cmic.midasgeneralsegmentor";

/**************************************************************
 * Start of Constructing/Destructing the View stuff.
 *************************************************************/

//-----------------------------------------------------------------------------
niftkGeneralSegmentorView::niftkGeneralSegmentorView()
: niftkBaseSegmentorView()
, m_ToolKeyPressStateMachine(NULL)
, m_GeneralSegmentorGUI(NULL)
, m_SliceNavigationController(NULL)
, m_SliceNavigationControllerObserverTag(0)
, m_FocusManagerObserverTag(0)
, m_IsUpdating(false)
, m_IsDeleting(false)
, m_IsChangingSlice(false)
, m_PreviousSliceNumber(0)
, m_IsRestarting(false)
{
  m_Interface = niftkGeneralSegmentorEventInterface::New();
  m_Interface->SetGeneralSegmentorView(this);
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
void niftkGeneralSegmentorView::CreateQtPartControl(QWidget *parent)
{
  niftkBaseSegmentorView::CreateQtPartControl(parent);

//    m_ToolKeyPressStateMachine = niftk::MIDASToolKeyPressStateMachine::New("MIDASToolKeyPressStateMachine", this);
  m_ToolKeyPressStateMachine = niftk::MIDASToolKeyPressStateMachine::New(this);
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorController* niftkGeneralSegmentorView::CreateSegmentorController()
{
  m_GeneralSegmentorController = new niftkGeneralSegmentorController(this);
  return m_GeneralSegmentorController;
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::Visible()
{
  niftkBaseSegmentorView::Visible();

  /// TODO
//  mitk::GlobalInteraction::GetInstance()->AddListener( m_ToolKeyPressStateMachine );

  // Connect registered tools back to here, so we can do seed processing logic here.
  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  niftk::MIDASPolyTool* midasPolyTool = dynamic_cast<niftk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASPolyTool>()));
  midasPolyTool->ContoursHaveChanged += mitk::MessageDelegate<niftkGeneralSegmentorView>( this, &niftkGeneralSegmentorView::OnContoursChanged );

  niftk::MIDASDrawTool* midasDrawTool = dynamic_cast<niftk::MIDASDrawTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASDrawTool>()));
  midasDrawTool->ContoursHaveChanged += mitk::MessageDelegate<niftkGeneralSegmentorView>( this, &niftkGeneralSegmentorView::OnContoursChanged );

}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::Hidden()
{
  niftkBaseSegmentorView::Hidden();

  if (m_SliceNavigationController.IsNotNull())
  {
    m_SliceNavigationController->RemoveObserver(m_SliceNavigationControllerObserverTag);
  }

  /// TODO
//  mitk::GlobalInteraction::GetInstance()->RemoveListener(m_ToolKeyPressStateMachine);

  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  niftk::MIDASPolyTool* midasPolyTool = dynamic_cast<niftk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASPolyTool>()));
  midasPolyTool->ContoursHaveChanged -= mitk::MessageDelegate<niftkGeneralSegmentorView>( this, &niftkGeneralSegmentorView::OnContoursChanged );

  niftk::MIDASDrawTool* midasDrawTool = dynamic_cast<niftk::MIDASDrawTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASDrawTool>()));
  midasDrawTool->ContoursHaveChanged -= mitk::MessageDelegate<niftkGeneralSegmentorView>( this, &niftkGeneralSegmentorView::OnContoursChanged );

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
mitk::DataNode::Pointer niftkGeneralSegmentorView::CreateHelperImage(mitk::Image::Pointer referenceImage, mitk::DataNode::Pointer segmentationNode, float r, float g, float b, std::string name, bool visible, int layer)
{
  assert(m_GeneralSegmentorController);
  return m_GeneralSegmentorController->CreateHelperImage(referenceImage, segmentationNode, r, g, b, name, visible, layer);
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer niftkGeneralSegmentorView::CreateContourSet(mitk::DataNode::Pointer segmentationNode, float r, float g, float b, std::string name, bool visible, int layer)
{
  assert(m_GeneralSegmentorController);
  return m_GeneralSegmentorController->CreateContourSet(segmentationNode, r, g, b, name, visible, layer);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnNewSegmentationButtonClicked()
{
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

  m_IsRestarting = isRestarting;

  // Finally, select the new segmentation node.
  this->SetCurrentSelection(newSegmentation);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::StoreInitialSegmentation()
{
  assert(m_GeneralSegmentorController);
  m_GeneralSegmentorController->StoreInitialSegmentation();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::onVisibilityChanged(const mitk::DataNode* node)
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  std::vector<mitk::DataNode*> workingData = this->GetWorkingData();
  if (!workingData.empty() && node == workingData[niftk::MIDASTool::SEGMENTATION])
  {
    bool segmentationNodeVisibility;
    if (node->GetVisibility(segmentationNodeVisibility, 0) && segmentationNodeVisibility)
    {
      workingData[niftk::MIDASTool::SEEDS]->SetVisibility(true);
      workingData[niftk::MIDASTool::CONTOURS]->SetVisibility(true);
      workingData[niftk::MIDASTool::DRAW_CONTOURS]->SetVisibility(true);
      if (m_GeneralSegmentorGUI->IsSeePriorCheckBoxChecked())
      {
        workingData[niftk::MIDASTool::PRIOR_CONTOURS]->SetVisibility(true);
      }
      if (m_GeneralSegmentorGUI->IsSeeNextCheckBoxChecked())
      {
        workingData[niftk::MIDASTool::NEXT_CONTOURS]->SetVisibility(true);
      }
      if (m_GeneralSegmentorGUI->IsThresholdingCheckBoxChecked())
      {
        workingData[niftk::MIDASTool::REGION_GROWING]->SetVisibility(true);
      }
      workingData[niftk::MIDASTool::INITIAL_SEGMENTATION]->SetVisibility(false);
      workingData[niftk::MIDASTool::INITIAL_SEEDS]->SetVisibility(false);

      mitk::ToolManager::Pointer toolManager = this->GetToolManager();
      niftk::MIDASPolyTool* polyTool = static_cast<niftk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASPolyTool>()));
      assert(polyTool);
      polyTool->SetFeedbackContourVisible(toolManager->GetActiveTool() == polyTool);
    }
    else
    {
      for (std::size_t i = 1; i < workingData.size(); ++i)
      {
        workingData[i]->SetVisibility(false);
      }
    }
  }
}


/**************************************************************
 * End of: Functions to create reference data (hidden nodes)
 *************************************************************/


/**************************************************************
 * Start of: Utility functions
 *************************************************************/

//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::EnableSegmentationWidgets(bool enabled)
{
  assert(m_GeneralSegmentorGUI);
  m_GeneralSegmentorGUI->SetAllWidgetsEnabled(enabled);
  bool thresholdingIsOn = m_GeneralSegmentorGUI->IsThresholdingCheckBoxChecked();
  m_GeneralSegmentorGUI->SetThresholdingWidgetsEnabled(thresholdingIsOn);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::RecalculateMinAndMaxOfImage()
{
  assert(m_GeneralSegmentorController);
  m_GeneralSegmentorController->RecalculateMinAndMaxOfImage();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::RecalculateMinAndMaxOfSeedValues()
{
  assert(m_GeneralSegmentorController);
  m_GeneralSegmentorController->RecalculateMinAndMaxOfSeedValues();
}


//-----------------------------------------------------------------------------
mitk::PointSet* niftkGeneralSegmentorView::GetSeeds()
{
  assert(m_GeneralSegmentorController);
  return m_GeneralSegmentorController->GetSeeds();
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorView::HasInitialisedWorkingData()
{
  assert(m_GeneralSegmentorController);
  return m_GeneralSegmentorController->HasInitialisedWorkingData();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::FilterSeedsToCurrentSlice(
    mitk::PointSet& inputPoints,
    int& axisNumber,
    int& sliceNumber,
    mitk::PointSet& outputPoints
    )
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    try
    {
      AccessFixedDimensionByItk_n(referenceImage,
          niftk::ITKFilterSeedsToCurrentSlice, 3,
          (inputPoints,
           axisNumber,
           sliceNumber,
           outputPoints
          )
        );
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so abandoning FilterSeedsToCurrentSlice, caused by:" << e.what();
    }
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::FilterSeedsToEnclosedSeedsOnCurrentSlice(
    mitk::PointSet& inputPoints,
    bool& thresholdOn,
    int& sliceNumber,
    mitk::PointSet& outputPoints
    )
{
  outputPoints.Clear();

  mitk::PointSet::Pointer singleSeedPointSet = mitk::PointSet::New();

  mitk::PointSet::PointsConstIterator inputPointsIt = inputPoints.Begin();
  mitk::PointSet::PointsConstIterator inputPointsEnd = inputPoints.End();
  for ( ; inputPointsIt != inputPointsEnd; ++inputPointsIt)
  {
    mitk::PointSet::PointType point = inputPointsIt->Value();
    mitk::PointSet::PointIdentifier pointID = inputPointsIt->Index();

    singleSeedPointSet->Clear();
    singleSeedPointSet->InsertPoint(0, point);

    bool unenclosed = this->DoesSliceHaveUnenclosedSeeds(thresholdOn, sliceNumber, *(singleSeedPointSet.GetPointer()));

    if (!unenclosed)
    {
      outputPoints.InsertPoint(pointID, point);
    }
  }
}


/**************************************************************
 * End of: Utility functions
 *************************************************************/

/**************************************************************
 * Start of: Functions for OK/Reset/Cancel/Close.
 * i.e. finishing a segmentation, and destroying stuff.
 *************************************************************/


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::DestroyPipeline()
{
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    m_IsDeleting = true;
    try
    {
      AccessFixedDimensionByItk(referenceImage, niftk::ITKDestroyPipeline, 3);
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so abandoning destroying the ITK pipeline, caused by:" << e.what();
    }
    m_IsDeleting = false;
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::RemoveWorkingData()
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  m_IsDeleting = true;

  mitk::ToolManager* toolManager = this->GetToolManager();
  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();

  // We don't do the first image, as thats the final segmentation.
  for (unsigned int i = 1; i < workingData.size(); i++)
  {
    this->GetDataStorage()->Remove(workingData[i]);
  }

  mitk::ToolManager::DataVectorType emptyWorkingDataArray;
  toolManager->SetWorkingData(emptyWorkingDataArray);
  toolManager->ActivateTool(-1);

  m_IsDeleting = false;
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::RestoreInitialSegmentation()
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::DataNode::Pointer segmentationNode = this->GetToolManager()->GetWorkingData(niftk::MIDASTool::SEGMENTATION);
  assert(segmentationNode);

  mitk::DataNode::Pointer seedsNode = this->GetToolManager()->GetWorkingData(niftk::MIDASTool::SEEDS);
  assert(seedsNode);

  try
  {
    /// Originally, this function cleared the segmentation and the pointset, but
    /// now we rather restore the initial state of the segmentation as it was
    /// when we pressed the Create/restart segmentation button.

//    mitk::Image::Pointer segmentationImage = dynamic_cast<mitk::Image*>(segmentationNode->GetData());
//    assert(segmentationImage);
//    AccessFixedDimensionByItk(segmentationImage.GetPointer(), niftk::ITKClearImage, 3);
//    segmentationImage->Modified();
//    segmentationNode->Modified();

//    mitk::PointSet::Pointer seeds = this->GetSeeds();
//    seeds->Clear();

    mitk::DataNode::Pointer initialSegmentationNode = this->GetToolManager()->GetWorkingData(niftk::MIDASTool::INITIAL_SEGMENTATION);
    mitk::DataNode::Pointer initialSeedsNode = this->GetToolManager()->GetWorkingData(niftk::MIDASTool::INITIAL_SEEDS);

    segmentationNode->SetData(dynamic_cast<mitk::Image*>(initialSegmentationNode->GetData())->Clone());
    seedsNode->SetData(dynamic_cast<mitk::PointSet*>(initialSeedsNode->GetData())->Clone());

    // This will cause OnSliceNumberChanged to be called, forcing refresh of all contours.
    if (m_SliceNavigationController)
    {
      m_SliceNavigationController->SendSlice();
    }
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Caught exception during niftk::ITKClearImage, caused by:" << e.what();
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnOKButtonClicked()
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  // Set the colour to that which the user selected in the first place.
  mitk::DataNode::Pointer workingData = this->GetToolManager()->GetWorkingData(niftk::MIDASTool::SEGMENTATION);
  workingData->SetProperty("color", workingData->GetProperty("midas.tmp.selectedcolor"));
  workingData->SetProperty("binaryimage.selectedcolor", workingData->GetProperty("midas.tmp.selectedcolor"));

  /// Apply the thresholds if we are thresholding, and chunk out the contour segments that
  /// do not close any region with seed.
  this->OnCleanButtonClicked();

  this->DestroyPipeline();
  this->RemoveWorkingData();
  this->EnableSegmentationWidgets(false);
  this->SetCurrentSelection(workingData);

  this->RequestRenderWindowUpdate();
  mitk::UndoController::GetCurrentUndoModel()->Clear();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnResetButtonClicked()
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  int returnValue = QMessageBox::warning(this->GetParent(), tr("NiftyView"),
                                                            tr("Clear all slices ? \n This is not Undo-able! \n Are you sure?"),
                                                            QMessageBox::Yes | QMessageBox::No);
  if (returnValue == QMessageBox::No)
  {
    return;
  }

  this->ClearWorkingData();
  this->UpdateRegionGrowing();
  this->UpdatePriorAndNext();
  this->UpdateCurrentSliceContours();
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnCancelButtonClicked()
{
  this->DiscardSegmentation();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::ClosePart()
{
  this->DiscardSegmentation();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::DiscardSegmentation()
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::DataNode::Pointer segmentationNode = this->GetToolManager()->GetWorkingData(niftk::MIDASTool::SEGMENTATION);
  assert(segmentationNode);

  this->DestroyPipeline();
  if (m_IsRestarting)
  {
    this->RestoreInitialSegmentation();
    this->RemoveWorkingData();
  }
  else
  {
    this->RemoveWorkingData();
    this->GetDataStorage()->Remove(segmentationNode);
  }
  this->EnableSegmentationWidgets(false);
  this->SetReferenceImageSelected();
  this->RequestRenderWindowUpdate();
  mitk::UndoController::GetCurrentUndoModel()->Clear();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnRestartButtonClicked()
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  int returnValue = QMessageBox::warning(this->GetParent(), tr("NiftyView"),
                                                            tr("Discard all changes?\nThis is not Undo-able!\nAre you sure?"),
                                                            QMessageBox::Yes | QMessageBox::No);
  if (returnValue == QMessageBox::No)
  {
    return;
  }

  this->RestoreInitialSegmentation();
  this->UpdateRegionGrowing();
  this->UpdatePriorAndNext();
  this->UpdateCurrentSliceContours();
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::ClearWorkingData()
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::DataNode::Pointer workingData = this->GetToolManager()->GetWorkingData(niftk::MIDASTool::SEGMENTATION);
  assert(workingData);

  mitk::Image::Pointer segmentationImage = dynamic_cast<mitk::Image*>(workingData->GetData());
  assert(segmentationImage);

  try
  {
    AccessFixedDimensionByItk(segmentationImage.GetPointer(), niftk::ITKClearImage, 3);
    segmentationImage->Modified();
    workingData->Modified();

    mitk::PointSet::Pointer seeds = this->GetSeeds();
    seeds->Clear();

    // This will cause OnSliceNumberChanged to be called, forcing refresh of all contours.
    if (m_SliceNavigationController)
    {
      m_SliceNavigationController->SendSlice();
    }
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Caught exception during niftk::ITKClearImage, caused by:" << e.what();
  }
}


/**************************************************************
 * End of: Functions for OK/Reset/Cancel/Close.
 *************************************************************/

/**************************************************************
 * Start of: Functions for simply tool toggling
 *************************************************************/

//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::ToggleTool(int toolId)
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  int activeToolId = toolManager->GetActiveToolID();

  if (toolId == activeToolId)
  {
    toolManager->ActivateTool(-1);
  }
  else
  {
    toolManager->ActivateTool(toolId);
  }
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorView::SelectSeedTool()
{
  /// Note:
  /// If the tool selection box is disabled then the tools are not registered to
  /// the tool manager ( RegisterClient() ). Then if you activate a tool and another
  /// tool was already active, then its interaction event observer service tries to
  /// be unregistered. But since the tools was not registered into the tool manager,
  /// the observer service is still null, and the attempt to unregister it causes crash.
  ///
  /// Consequence:
  /// We should not do anything with the tools until they are registered to the
  /// tool manager.

  if (m_GeneralSegmentorGUI->IsToolSelectorEnabled())
  {
    mitk::ToolManager* toolManager = this->GetToolManager();
    int activeToolId = toolManager->GetActiveToolID();
    int seedToolId = toolManager->GetToolIdByToolType<niftk::MIDASSeedTool>();

    if (seedToolId != activeToolId)
    {
      toolManager->ActivateTool(seedToolId);
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorView::SelectDrawTool()
{
  /// Note: see comment in SelectSeedTool().
  if (m_GeneralSegmentorGUI->IsToolSelectorEnabled())
  {
    mitk::ToolManager* toolManager = this->GetToolManager();
    int activeToolId = toolManager->GetActiveToolID();
    int drawToolId = toolManager->GetToolIdByToolType<niftk::MIDASDrawTool>();

    if (drawToolId != activeToolId)
    {
      toolManager->ActivateTool(drawToolId);
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorView::SelectPolyTool()
{
  /// Note: see comment in SelectSeedTool().
  if (m_GeneralSegmentorGUI->IsToolSelectorEnabled())
  {
    mitk::ToolManager* toolManager = this->GetToolManager();
    int activeToolId = toolManager->GetActiveToolID();
    int polyToolId = toolManager->GetToolIdByToolType<niftk::MIDASPolyTool>();

    if (polyToolId != activeToolId)
    {
      toolManager->ActivateTool(polyToolId);
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorView::UnselectTools()
{
  if (m_GeneralSegmentorGUI->IsToolSelectorEnabled())
  {
    mitk::ToolManager* toolManager = this->GetToolManager();

    if (toolManager->GetActiveToolID() != -1)
    {
      toolManager->ActivateTool(-1);
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorView::SelectViewMode()
{
  /// Note: see comment in SelectSeedTool().
  if (m_GeneralSegmentorGUI->IsToolSelectorEnabled())
  {
    if (!this->HasInitialisedWorkingData())
    {
      QList<mitk::DataNode::Pointer> selectedNodes = this->GetDataManagerSelection();
      foreach (mitk::DataNode::Pointer selectedNode, selectedNodes)
      {
        selectedNode->SetVisibility(!selectedNode->IsVisible(0));
      }
      this->RequestRenderWindowUpdate();

      return true;
    }

    mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
    bool segmentationNodeIsVisible = workingData[niftk::MIDASTool::SEGMENTATION]->IsVisible(0);
    workingData[niftk::MIDASTool::SEGMENTATION]->SetVisibility(!segmentationNodeIsVisible);
    this->RequestRenderWindowUpdate();

    return true;
  }

  return false;
}


/**************************************************************
 * End of: Functions for simply tool toggling
 *************************************************************/

/**************************************************************
 * Start of: The main MIDAS business logic.
 *************************************************************/

//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::InitialiseSeedsForWholeVolume()
{
  assert(m_GeneralSegmentorController);
  m_GeneralSegmentorController->InitialiseSeedsForWholeVolume();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnFocusChanged()
{
  QmitkBaseView::OnFocusChanged();

  mitk::BaseRenderer* focusedRenderer = this->GetFocusedRenderer();

  if (focusedRenderer != NULL)
  {

    if (m_SliceNavigationController.IsNotNull())
    {
      m_SliceNavigationController->RemoveObserver(m_SliceNavigationControllerObserverTag);
    }

    m_SliceNavigationController = this->GetSliceNavigationController();

    if (m_SliceNavigationController.IsNotNull())
    {
      itk::ReceptorMemberCommand<niftkGeneralSegmentorView>::Pointer onSliceChangedCommand =
        itk::ReceptorMemberCommand<niftkGeneralSegmentorView>::New();

      onSliceChangedCommand->SetCallbackFunction( this, &niftkGeneralSegmentorView::OnSliceChanged );


      m_PreviousSliceNumber = -1;
      m_PreviousFocusPoint.Fill(0);
      m_CurrentFocusPoint.Fill(0);

      m_SliceNavigationControllerObserverTag =
          m_SliceNavigationController->AddObserver(
              mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onSliceChangedCommand);

      m_SliceNavigationController->SendSlice();
    }

    this->UpdatePriorAndNext();
    this->OnThresholdingCheckBoxToggled(m_GeneralSegmentorGUI->IsThresholdingCheckBoxChecked());
    this->RequestRenderWindowUpdate();
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::UpdateCurrentSliceContours(bool updateRendering)
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
  int axisNumber = this->GetViewAxis();

  mitk::Image::Pointer workingImage = this->GetWorkingImageFromToolManager(0);
  assert(workingImage);

  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
  mitk::ContourModelSet::Pointer contourSet = dynamic_cast<mitk::ContourModelSet*>(workingData[niftk::MIDASTool::CONTOURS]->GetData());

  // TODO
  // This assertion fails sometimes if both the morphological and irregular (this) volume editor is
  // switched on and you are using the paintbrush tool of the morpho editor.
//  assert(contourSet);

  if (contourSet)
  {
    if (sliceNumber >= 0 && axisNumber >= 0)
    {
      niftk::GenerateOutlineFromBinaryImage(workingImage, axisNumber, sliceNumber, sliceNumber, contourSet);

      if (contourSet->GetSize() > 0)
      {
        workingData[niftk::MIDASTool::CONTOURS]->Modified();

        if (updateRendering)
        {
          this->RequestRenderWindowUpdate();
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorView::DoesSliceHaveUnenclosedSeeds(bool thresholdOn, int sliceNumber)
{
  mitk::PointSet* seeds = this->GetSeeds();
  assert(seeds);

  return this->DoesSliceHaveUnenclosedSeeds(thresholdOn, sliceNumber, *seeds);
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorView::DoesSliceHaveUnenclosedSeeds(bool thresholdOn, int sliceNumber, mitk::PointSet& seeds)
{
  bool sliceDoesHaveUnenclosedSeeds = false;

  if (!this->HasInitialisedWorkingData())
  {
    return sliceDoesHaveUnenclosedSeeds;
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  mitk::Image::Pointer segmentationImage = this->GetWorkingImageFromToolManager(0);

  mitk::ToolManager *toolManager = this->GetToolManager();
  assert(toolManager);

  niftk::MIDASPolyTool *polyTool = static_cast<niftk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASPolyTool>()));
  assert(polyTool);

  mitk::ContourModelSet::Pointer polyToolContours = mitk::ContourModelSet::New();
  mitk::ContourModel* polyToolContour = polyTool->GetContour();
  if (polyToolContour != NULL && polyToolContour->GetNumberOfVertices() >= 2)
  {
    polyToolContours->AddContourModel(polyToolContour);
  }

  mitk::ContourModelSet* segmentationContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[niftk::MIDASTool::CONTOURS]->GetData());
  mitk::ContourModelSet* drawToolContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[niftk::MIDASTool::DRAW_CONTOURS]->GetData());

  double lowerThreshold = m_GeneralSegmentorGUI->GetLowerThreshold();
  double upperThreshold = m_GeneralSegmentorGUI->GetUpperThreshold();

  int axisNumber = this->GetViewAxis();

  if (axisNumber != -1 && sliceNumber != -1)
  {
    try
    {
      AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
        niftk::ITKSliceDoesHaveUnEnclosedSeeds, 3,
          (seeds,
           *segmentationContours,
           *polyToolContours,
           *drawToolContours,
           *segmentationImage,
            lowerThreshold,
            upperThreshold,
            thresholdOn,
            axisNumber,
            sliceNumber,
            sliceDoesHaveUnenclosedSeeds
          )
      );
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception during niftk::ITKSliceDoesHaveUnEnclosedSeeds, so will return false, caused by:" << e.what();
    }
  }

  return sliceDoesHaveUnenclosedSeeds;
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorView::CleanSlice()
{
  /// Note: see comment in SelectSeedTool().
  if (m_GeneralSegmentorGUI->IsToolSelectorEnabled())
  {
    this->OnCleanButtonClicked();
    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnSeePriorCheckBoxToggled(bool checked)
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();

  if (checked)
  {
    this->UpdatePriorAndNext();
  }
  workingData[niftk::MIDASTool::PRIOR_CONTOURS]->SetVisibility(checked);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnSeeNextCheckBoxToggled(bool checked)
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();

  if (checked)
  {
    this->UpdatePriorAndNext();
  }
  workingData[niftk::MIDASTool::NEXT_CONTOURS]->SetVisibility(checked);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::UpdatePriorAndNext(bool updateRendering)
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
  int axisNumber = this->GetViewAxis();

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
  mitk::Image::Pointer segmentationImage = this->GetWorkingImageFromToolManager(0);

  if (m_GeneralSegmentorGUI->IsSeePriorCheckBoxChecked())
  {
    mitk::ContourModelSet::Pointer contourSet = dynamic_cast<mitk::ContourModelSet*>(workingData[niftk::MIDASTool::PRIOR_CONTOURS]->GetData());
    niftk::GenerateOutlineFromBinaryImage(segmentationImage, axisNumber, sliceNumber-1, sliceNumber, contourSet);

    if (contourSet->GetSize() > 0)
    {
      workingData[niftk::MIDASTool::PRIOR_CONTOURS]->Modified();

      if (updateRendering)
      {
        this->RequestRenderWindowUpdate();
      }
    }
  }

  if (m_GeneralSegmentorGUI->IsSeeNextCheckBoxChecked())
  {
    mitk::ContourModelSet::Pointer contourSet = dynamic_cast<mitk::ContourModelSet*>(workingData[niftk::MIDASTool::NEXT_CONTOURS]->GetData());
    niftk::GenerateOutlineFromBinaryImage(segmentationImage, axisNumber, sliceNumber+1, sliceNumber, contourSet);

    if (contourSet->GetSize() > 0)
    {
      workingData[niftk::MIDASTool::NEXT_CONTOURS]->Modified();

      if (updateRendering)
      {
        this->RequestRenderWindowUpdate();
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnThresholdingCheckBoxToggled(bool checked)
{
  if (!this->HasInitialisedWorkingData())
  {
    // So, if there is NO working data, we leave the widgets disabled regardless.
    m_GeneralSegmentorGUI->SetThresholdingWidgetsEnabled(false);
    return;
  }

  this->RecalculateMinAndMaxOfImage();
  this->RecalculateMinAndMaxOfSeedValues();

  m_GeneralSegmentorGUI->SetThresholdingWidgetsEnabled(checked);

  if (checked)
  {
    this->UpdateRegionGrowing();
  }

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
  workingData[niftk::MIDASTool::REGION_GROWING]->SetVisibility(checked);

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnThresholdValueChanged()
{
  this->UpdateRegionGrowing();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::UpdateRegionGrowing(bool updateRendering)
{
  bool isVisible = m_GeneralSegmentorGUI->IsThresholdingCheckBoxChecked();
  int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
  double lowerThreshold = m_GeneralSegmentorGUI->GetLowerThreshold();
  double upperThreshold = m_GeneralSegmentorGUI->GetUpperThreshold();
  bool skipUpdate = !isVisible;

  if (isVisible)
  {
    this->UpdateRegionGrowing(isVisible, sliceNumber, lowerThreshold, upperThreshold, skipUpdate);

    if (updateRendering)
    {
      this->RequestRenderWindowUpdate();
    }
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::UpdateRegionGrowing(
    bool isVisible,
    int sliceNumber,
    double lowerThreshold,
    double upperThreshold,
    bool skipUpdate
    )
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    mitk::DataNode::Pointer segmentationNode = this->GetWorkingData()[niftk::MIDASTool::SEGMENTATION];
    mitk::Image::Pointer segmentationImage = this->GetWorkingImageFromToolManager(niftk::MIDASTool::SEGMENTATION);

    if (segmentationImage.IsNotNull() && segmentationNode.IsNotNull())
    {

      mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
      workingData[niftk::MIDASTool::REGION_GROWING]->SetVisibility(isVisible);

      m_IsUpdating = true;

      mitk::DataNode::Pointer regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(niftk::MIDASTool::REGION_GROWING_NAME.c_str(), segmentationNode, true);
      assert(regionGrowingNode);

      mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
      assert(regionGrowingImage);

      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      mitk::ToolManager *toolManager = this->GetToolManager();
      assert(toolManager);

      niftk::MIDASPolyTool *polyTool = static_cast<niftk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASPolyTool>()));
      assert(polyTool);

      mitk::ContourModelSet::Pointer polyToolContours = mitk::ContourModelSet::New();

      mitk::ContourModel* polyToolContour = polyTool->GetContour();
      if (polyToolContour != NULL && polyToolContour->GetNumberOfVertices() >= 2)
      {
        polyToolContours->AddContourModel(polyToolContour);
      }

      mitk::ContourModelSet* segmentationContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[niftk::MIDASTool::CONTOURS]->GetData());
      mitk::ContourModelSet* drawToolContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[niftk::MIDASTool::DRAW_CONTOURS]->GetData());

      int axisNumber = this->GetViewAxis();

      if (axisNumber != -1 && sliceNumber != -1)
      {
        try
        {
          AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
              niftk::ITKUpdateRegionGrowing, 3,
              (skipUpdate,
               *segmentationImage,
               *seeds,
               *segmentationContours,
               *drawToolContours,
               *polyToolContours,
               sliceNumber,
               axisNumber,
               lowerThreshold,
               upperThreshold,
               regionGrowingNode,  // This is the node for the image we are writing to.
               regionGrowingImage  // This is the image we are writing to.
              )
            );

          regionGrowingImage->Modified();
          regionGrowingNode->Modified();
        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Could not do region growing: Caught exception, so abandoning ITK pipeline update:" << e.what();
        }
      }
      else
      {
        MITK_ERROR << "Could not do region growing: Error axisNumber=" << axisNumber << ", sliceNumber=" << sliceNumber << std::endl;
      }

      m_IsUpdating = false;

    }
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnThresholdApplyButtonClicked()
{
  int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
  this->DoThresholdApply(sliceNumber, sliceNumber, true, false, false);
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorView::DoThresholdApply(
    int oldSliceNumber,
    int newSliceNumber,
    bool optimiseSeeds,
    bool newSliceEmpty,
    bool newCheckboxStatus)
{
  if (!this->HasInitialisedWorkingData())
  {
    return false;
  }

  bool updateWasApplied = false;

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    mitk::DataNode::Pointer segmentationNode = this->GetWorkingData()[niftk::MIDASTool::SEGMENTATION];
    mitk::Image::Pointer segmentationImage = this->GetWorkingImageFromToolManager(niftk::MIDASTool::SEGMENTATION);

    if (segmentationImage.IsNotNull() && segmentationNode.IsNotNull())
    {
      mitk::DataNode::Pointer regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(niftk::MIDASTool::REGION_GROWING_NAME.c_str(), segmentationNode, true);
      assert(regionGrowingNode);

      mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
      assert(regionGrowingImage);

      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      mitk::ToolManager *toolManager = this->GetToolManager();
      assert(toolManager);

      niftk::MIDASDrawTool *drawTool = static_cast<niftk::MIDASDrawTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASDrawTool>()));
      assert(drawTool);

      int axisNumber = this->GetViewAxis();

      mitk::PointSet::Pointer copyOfInputSeeds = mitk::PointSet::New();
      mitk::PointSet::Pointer outputSeeds = mitk::PointSet::New();
      std::vector<int> outputRegion;

      if (axisNumber != -1 && oldSliceNumber != -1)
      {
        m_IsUpdating = true;

        try
        {
          AccessFixedDimensionByItk_n(regionGrowingImage,
              niftk::ITKPreProcessingOfSeedsForChangingSlice, 3,
              (*seeds,
               oldSliceNumber,
               axisNumber,
               newSliceNumber,
               optimiseSeeds,
               newSliceEmpty,
               *(copyOfInputSeeds.GetPointer()),
               *(outputSeeds.GetPointer()),
               outputRegion
              )
            );

          bool currentCheckboxStatus = m_GeneralSegmentorGUI->IsThresholdingCheckBoxChecked();

          if (toolManager->GetActiveToolID() == toolManager->GetToolIdByToolType<niftk::MIDASPolyTool>())
          {
            toolManager->ActivateTool(-1);
          }

          mitk::UndoStackItem::IncCurrObjectEventId();
          mitk::UndoStackItem::IncCurrGroupEventId();
          mitk::UndoStackItem::ExecuteIncrement();

          QString message = tr("Apply threshold on slice %1").arg(oldSliceNumber);
          niftk::OpThresholdApply::ProcessorPointer processor = niftk::OpThresholdApply::ProcessorType::New();
          niftk::OpThresholdApply *doThresholdOp = new niftk::OpThresholdApply(niftk::OP_THRESHOLD_APPLY, true, outputRegion, processor, newCheckboxStatus);
          niftk::OpThresholdApply *undoThresholdOp = new niftk::OpThresholdApply(niftk::OP_THRESHOLD_APPLY, false, outputRegion, processor, currentCheckboxStatus);
          mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doThresholdOp, undoThresholdOp, message.toStdString());
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
          this->ExecuteOperation(doThresholdOp);

          message = tr("Propagate seeds on slice %1").arg(oldSliceNumber);
          niftk::OpPropagateSeeds *doPropOp = new niftk::OpPropagateSeeds(niftk::OP_PROPAGATE_SEEDS, true, newSliceNumber, axisNumber, outputSeeds);
          niftk::OpPropagateSeeds *undoPropOp = new niftk::OpPropagateSeeds(niftk::OP_PROPAGATE_SEEDS, false, oldSliceNumber, axisNumber, copyOfInputSeeds);
          mitk::OperationEvent* operationPropEvent = new mitk::OperationEvent( m_Interface, doPropOp, undoPropOp, message.toStdString());
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationPropEvent );
          this->ExecuteOperation(doPropOp);

          drawTool->ClearWorkingData();

          bool updateRendering(false);
          this->UpdatePriorAndNext(updateRendering);
          this->UpdateRegionGrowing(updateRendering);
          this->UpdateCurrentSliceContours(updateRendering);

          updateWasApplied = true;
        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Could not do threshold apply command: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        }
        catch( itk::ExceptionObject &err )
        {
          MITK_ERROR << "Could not do threshold apply command: Caught itk::ExceptionObject:" << err.what() << std::endl;
        }

        m_IsUpdating = false;

      } // end if we have valid axis / slice
    } // end if we have working data
  }// end if we have a reference image

  if (updateWasApplied)
  {
    this->RequestRenderWindowUpdate();
  }
  return updateWasApplied;
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnSliceChanged(const itk::EventObject & geometrySliceEvent)
{
  mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();
  if (renderWindowPart != NULL &&  !m_IsChangingSlice)
  {
    int previousSlice = m_PreviousSliceNumber;

    int currentSlice = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
    mitk::Point3D currentFocus = renderWindowPart->GetSelectedPosition();

    if (previousSlice == -1)
    {
      previousSlice = currentSlice;
      m_PreviousFocusPoint = currentFocus;
      m_CurrentFocusPoint = currentFocus;
    }

    this->OnSliceNumberChanged(previousSlice, currentSlice);

    m_PreviousSliceNumber = currentSlice;
    m_PreviousFocusPoint = currentFocus;
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnSliceNumberChanged(int beforeSliceNumber, int afterSliceNumber)
{
  if (  !this->HasInitialisedWorkingData()
      || m_IsUpdating
      || m_IsChangingSlice
      || beforeSliceNumber == -1
      || afterSliceNumber == -1
      || abs(beforeSliceNumber - afterSliceNumber) != 1
      )
  {
    m_PreviousSliceNumber = afterSliceNumber;
    m_PreviousFocusPoint = m_CurrentFocusPoint;

    bool updateRendering(false);
    this->UpdateCurrentSliceContours(updateRendering);
    this->UpdateRegionGrowing(updateRendering);
    this->RequestRenderWindowUpdate();

    return;
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    mitk::DataNode::Pointer segmentationNode = this->GetWorkingData()[niftk::MIDASTool::SEGMENTATION];
    mitk::Image::Pointer segmentationImage = this->GetWorkingImageFromToolManager(niftk::MIDASTool::SEGMENTATION);

    if (segmentationNode.IsNotNull() && segmentationImage.IsNotNull())
    {
      int axisNumber = this->GetViewAxis();
      MIDASOrientation tmpOrientation = this->GetOrientationAsEnum();
      itk::Orientation orientation = niftk::GetItkOrientation(tmpOrientation);

      mitk::ToolManager *toolManager = this->GetToolManager();
      assert(toolManager);

      niftk::MIDASDrawTool *drawTool = static_cast<niftk::MIDASDrawTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASDrawTool>()));
      assert(drawTool);

      if (   axisNumber != -1
          && beforeSliceNumber != -1
          && afterSliceNumber != -1
          && beforeSliceNumber != afterSliceNumber)
      {
        std::vector<int> outputRegion;
        mitk::PointSet::Pointer copyOfCurrentSeeds = mitk::PointSet::New();
        mitk::PointSet::Pointer propagatedSeeds = mitk::PointSet::New();
        mitk::PointSet* seeds = this->GetSeeds();
        bool nextSliceIsEmpty(true);
        bool thisSliceIsEmpty(false);

        m_IsUpdating = true;

        try
        {
          ///////////////////////////////////////////////////////
          // See: https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK/issues/1742
          //      for the whole logic surrounding changing slice.
          ///////////////////////////////////////////////////////

          AccessFixedDimensionByItk_n(segmentationImage,
              niftk::ITKSliceIsEmpty, 3,
              (axisNumber,
               afterSliceNumber,
               nextSliceIsEmpty
              )
            );

          if (m_GeneralSegmentorGUI->IsRetainMarksCheckBoxChecked())
          {
            int returnValue(QMessageBox::NoButton);

            if (!m_GeneralSegmentorGUI->IsThresholdingCheckBoxChecked())
            {
              AccessFixedDimensionByItk_n(segmentationImage,
                  niftk::ITKSliceIsEmpty, 3,
                  (axisNumber,
                   beforeSliceNumber,
                   thisSliceIsEmpty
                  )
                );
            }

            if (thisSliceIsEmpty)
            {
              returnValue = QMessageBox::warning(this->GetParent(), tr("NiftyView"),
                                                      tr("The current slice is empty - retain marks cannot be performed.\n"
                                                         "Use the 'wipe' functionality to erase slices instead"),
                                                      QMessageBox::Ok
                                   );
            }
            else if (!nextSliceIsEmpty)
            {
              returnValue = QMessageBox::warning(this->GetParent(), tr("NiftyView"),
                                                      tr("The new slice is not empty - retain marks will overwrite the slice.\n"
                                                         "Are you sure?"),
                                                      QMessageBox::Yes | QMessageBox::No);
            }

            if (returnValue == QMessageBox::Ok || returnValue == QMessageBox::No )
            {
              m_IsUpdating = false;
              m_PreviousSliceNumber = afterSliceNumber;
              m_PreviousFocusPoint = m_CurrentFocusPoint;
              this->UpdatePriorAndNext();
              this->UpdateRegionGrowing();
              this->UpdateCurrentSliceContours();
              this->RequestRenderWindowUpdate();

              return;
            }

            AccessFixedDimensionByItk_n(segmentationImage,
                niftk::ITKPreProcessingOfSeedsForChangingSlice, 3,
                (*seeds,
                 beforeSliceNumber,
                 axisNumber,
                 afterSliceNumber,
                 false, // We propagate seeds at current position, so no optimisation
                 nextSliceIsEmpty,
                 *(copyOfCurrentSeeds.GetPointer()),
                 *(propagatedSeeds.GetPointer()),
                 outputRegion
                )
              );

            if (m_GeneralSegmentorGUI->IsThresholdingCheckBoxChecked())
            {
              QString message = tr("Thresholding slice %1 before copying marks to slice %2").arg(beforeSliceNumber).arg(afterSliceNumber);
              niftk::OpThresholdApply::ProcessorPointer processor = niftk::OpThresholdApply::ProcessorType::New();
              niftk::OpThresholdApply *doThresholdOp = new niftk::OpThresholdApply(niftk::OP_THRESHOLD_APPLY, true, outputRegion, processor, true);
              niftk::OpThresholdApply *undoThresholdOp = new niftk::OpThresholdApply(niftk::OP_THRESHOLD_APPLY, false, outputRegion, processor, true);
              mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doThresholdOp, undoThresholdOp, message.toStdString());
              mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
              this->ExecuteOperation(doThresholdOp);

              drawTool->ClearWorkingData();
              this->UpdateCurrentSliceContours();
            }

            // Do retain marks, which copies slice from beforeSliceNumber to afterSliceNumber
            QString message = tr("Retaining marks in slice %1 and copying to %2").arg(beforeSliceNumber).arg(afterSliceNumber);
            niftk::OpRetainMarks::ProcessorPointer processor = niftk::OpRetainMarks::ProcessorType::New();
            niftk::OpRetainMarks *doOp = new niftk::OpRetainMarks(niftk::OP_RETAIN_MARKS, true, beforeSliceNumber, afterSliceNumber, axisNumber, orientation, outputRegion, processor);
            niftk::OpRetainMarks *undoOp = new niftk::OpRetainMarks(niftk::OP_RETAIN_MARKS, false, beforeSliceNumber, afterSliceNumber, axisNumber, orientation, outputRegion, processor);
            mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, message.toStdString());
            mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
            this->ExecuteOperation(doOp);
          }
          else // so, "Retain Marks" is Off.
          {
            AccessFixedDimensionByItk_n(segmentationImage,
                niftk::ITKPreProcessingOfSeedsForChangingSlice, 3,
                (*seeds,
                 beforeSliceNumber,
                 axisNumber,
                 afterSliceNumber,
                 true, // optimise seed position on current slice.
                 nextSliceIsEmpty,
                 *(copyOfCurrentSeeds.GetPointer()),
                 *(propagatedSeeds.GetPointer()),
                 outputRegion
                )
              );

            if (m_GeneralSegmentorGUI->IsThresholdingCheckBoxChecked())
            {
              niftk::OpThresholdApply::ProcessorPointer processor = niftk::OpThresholdApply::ProcessorType::New();
              niftk::OpThresholdApply *doApplyOp = new niftk::OpThresholdApply(niftk::OP_THRESHOLD_APPLY, true, outputRegion, processor, m_GeneralSegmentorGUI->IsThresholdingCheckBoxChecked());
              niftk::OpThresholdApply *undoApplyOp = new niftk::OpThresholdApply(niftk::OP_THRESHOLD_APPLY, false, outputRegion, processor, m_GeneralSegmentorGUI->IsThresholdingCheckBoxChecked());
              mitk::OperationEvent* operationApplyEvent = new mitk::OperationEvent( m_Interface, doApplyOp, undoApplyOp, "Apply threshold");
              mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationApplyEvent );
              this->ExecuteOperation(doApplyOp);

              drawTool->ClearWorkingData();
              this->UpdateCurrentSliceContours();
            }
            else // threshold box not checked
            {
              bool thisSliceHasUnenclosedSeeds = this->DoesSliceHaveUnenclosedSeeds(false, beforeSliceNumber);

              if (thisSliceHasUnenclosedSeeds)
              {
                niftk::OpWipe::ProcessorPointer processor = niftk::OpWipe::ProcessorType::New();
                niftk::OpWipe *doWipeOp = new niftk::OpWipe(niftk::OP_WIPE, true, beforeSliceNumber, axisNumber, outputRegion, propagatedSeeds, processor);
                niftk::OpWipe *undoWipeOp = new niftk::OpWipe(niftk::OP_WIPE, false, beforeSliceNumber, axisNumber, outputRegion, copyOfCurrentSeeds, processor);
                mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doWipeOp, undoWipeOp, "Wipe command");
                mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
                this->ExecuteOperation(doWipeOp);
              }
              else // so, we don't have unenclosed seeds
              {
                // There may be the case where the user has simply drawn a region, and put a seed in the middle.
                // So, we do a region growing, without intensity limits. (we already know there are no unenclosed seeds).

                this->UpdateRegionGrowing(false,
                                          beforeSliceNumber,
                                          referenceImage->GetStatistics()->GetScalarValueMinNoRecompute(),
                                          referenceImage->GetStatistics()->GetScalarValueMaxNoRecompute(),
                                          false);

                // Then we "apply" this region growing.
                niftk::OpThresholdApply::ProcessorPointer processor = niftk::OpThresholdApply::ProcessorType::New();
                niftk::OpThresholdApply *doApplyOp = new niftk::OpThresholdApply(niftk::OP_THRESHOLD_APPLY, true, outputRegion, processor, false);
                niftk::OpThresholdApply *undoApplyOp = new niftk::OpThresholdApply(niftk::OP_THRESHOLD_APPLY, false, outputRegion, processor, false);
                mitk::OperationEvent* operationApplyEvent = new mitk::OperationEvent( m_Interface, doApplyOp, undoApplyOp, "Apply threshold");
                mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationApplyEvent );
                this->ExecuteOperation(doApplyOp);

                drawTool->ClearWorkingData();

              } // end if/else unenclosed seeds
            } // end if/else thresholding on
          } // end if/else retain marks.

          mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();
          if (renderWindowPart != NULL)
          {
            m_CurrentFocusPoint = renderWindowPart->GetSelectedPosition();
          }

          QString message = tr("Propagate seeds from slice %1 to %2").arg(beforeSliceNumber).arg(afterSliceNumber);
          niftk::OpPropagateSeeds *doPropOp = new niftk::OpPropagateSeeds(niftk::OP_PROPAGATE_SEEDS, true, afterSliceNumber, axisNumber, propagatedSeeds);
          niftk::OpPropagateSeeds *undoPropOp = new niftk::OpPropagateSeeds(niftk::OP_PROPAGATE_SEEDS, false, beforeSliceNumber, axisNumber, copyOfCurrentSeeds);
          mitk::OperationEvent* operationPropEvent = new mitk::OperationEvent( m_Interface, doPropOp, undoPropOp, message.toStdString());
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationPropEvent );
          this->ExecuteOperation(doPropOp);

          message = tr("Change slice from %1 to %2").arg(beforeSliceNumber).arg(afterSliceNumber);
          niftk::OpChangeSliceCommand *doOp = new niftk::OpChangeSliceCommand(niftk::OP_CHANGE_SLICE, true, beforeSliceNumber, afterSliceNumber, m_PreviousFocusPoint, m_CurrentFocusPoint);
          niftk::OpChangeSliceCommand *undoOp = new niftk::OpChangeSliceCommand(niftk::OP_CHANGE_SLICE, false, beforeSliceNumber, afterSliceNumber, m_PreviousFocusPoint, m_CurrentFocusPoint);
          mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, message.toStdString());
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
          this->ExecuteOperation(doOp);
        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Could not change slice: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        }
        catch( itk::ExceptionObject &err )
        {
          MITK_ERROR << "Could not change slice: Caught itk::ExceptionObject:" << err.what() << std::endl;
        }

        m_IsUpdating = false;

        if (niftk::MIDASPolyTool* polyTool = dynamic_cast<niftk::MIDASPolyTool*>(toolManager->GetActiveTool()))
        {
//          toolManager->ActivateTool(-1);
          /// This makes the poly tool save its result to the working data nodes and stay it open.
          polyTool->Deactivated();
          polyTool->Activated();
        }

        bool updateRendering(false);
        this->UpdatePriorAndNext(updateRendering);
        this->UpdateRegionGrowing(updateRendering);
        this->UpdateCurrentSliceContours(updateRendering);
        this->RequestRenderWindowUpdate();

      } // end if, slice number, axis ok.
    } // end have working image
  } // end have reference image
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnCleanButtonClicked()
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  bool thresholdCheckBox = m_GeneralSegmentorGUI->IsThresholdingCheckBoxChecked();
  int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();

  if (!thresholdCheckBox)
  {
    bool hasUnenclosedSeeds = this->DoesSliceHaveUnenclosedSeeds(thresholdCheckBox, sliceNumber);
    if (hasUnenclosedSeeds)
    {
      int returnValue = QMessageBox::warning(this->GetParent(), tr("NiftyView"),
                                                       tr("There are unenclosed seeds - slice will be wiped\n"
                                                          "Are you sure?"),
                                                       QMessageBox::Yes | QMessageBox::No);
      if (returnValue == QMessageBox::Yes)
      {
        this->DoWipe(0);
      }
      return;
    }
  }

  bool cleanWasPerformed = false;

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    mitk::DataNode::Pointer segmentationNode = this->GetWorkingData()[niftk::MIDASTool::SEGMENTATION];
    mitk::Image::Pointer segmentationImage = this->GetWorkingImageFromToolManager(niftk::MIDASTool::SEGMENTATION);

    if (segmentationImage.IsNotNull() && segmentationNode.IsNotNull())
    {
      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      mitk::ToolManager *toolManager = this->GetToolManager();
      assert(toolManager);

      niftk::MIDASPolyTool *polyTool = static_cast<niftk::MIDASPolyTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASPolyTool>()));
      assert(polyTool);

      niftk::MIDASDrawTool *drawTool = static_cast<niftk::MIDASDrawTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASDrawTool>()));
      assert(drawTool);

      mitk::ContourModelSet::Pointer polyToolContours = mitk::ContourModelSet::New();

      mitk::ContourModel* polyToolContour = polyTool->GetContour();
      if (polyToolContour != NULL && polyToolContour->GetNumberOfVertices() >= 2)
      {
        polyToolContours->AddContourModel(polyToolContour);
      }

      mitk::ContourModelSet* segmentationContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[niftk::MIDASTool::CONTOURS]->GetData());
      assert(segmentationContours);

      mitk::ContourModelSet* drawToolContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[niftk::MIDASTool::DRAW_CONTOURS]->GetData());
      assert(drawToolContours);

      mitk::DataNode::Pointer regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(niftk::MIDASTool::REGION_GROWING_NAME.c_str(), segmentationNode, true);
      assert(regionGrowingNode);

      mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
      assert(regionGrowingImage);

      double lowerThreshold = m_GeneralSegmentorGUI->GetLowerThreshold();
      double upperThreshold = m_GeneralSegmentorGUI->GetUpperThreshold();
      int axisNumber = this->GetViewAxis();

      mitk::ContourModelSet::Pointer copyOfInputContourSet = mitk::ContourModelSet::New();
      mitk::ContourModelSet::Pointer outputContourSet = mitk::ContourModelSet::New();

      if (axisNumber != -1 && sliceNumber != -1)
      {
        m_IsUpdating = true;

        try
        {
          // Calculate the region of interest for this slice.
          std::vector<int> outputRegion;
          AccessFixedDimensionByItk_n(segmentationImage,
              niftk::ITKCalculateSliceRegionAsVector, 3,
              (axisNumber,
               sliceNumber,
               outputRegion
              )
            );

          if (thresholdCheckBox)
          {
            bool useThresholdsWhenCalculatingEnclosedSeeds = false;

            this->DoThresholdApply(sliceNumber, sliceNumber, true, false, true);

            // Get seeds just on the current slice
            mitk::PointSet::Pointer seedsForCurrentSlice = mitk::PointSet::New();
            this->FilterSeedsToCurrentSlice(
                *seeds,
                axisNumber,
                sliceNumber,
                *(seedsForCurrentSlice.GetPointer())
                );

            // Reduce the list just down to those that are fully enclosed.
            mitk::PointSet::Pointer enclosedSeeds = mitk::PointSet::New();
            this->FilterSeedsToEnclosedSeedsOnCurrentSlice(
                *seedsForCurrentSlice,
                useThresholdsWhenCalculatingEnclosedSeeds,
                sliceNumber,
                *(enclosedSeeds.GetPointer())
                );

            // Do region growing, using only enclosed seeds.
            AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
                niftk::ITKUpdateRegionGrowing, 3,
                (false,
                 *segmentationImage,
                 *enclosedSeeds,
                 *segmentationContours,
                 *drawToolContours,
                 *polyToolContours,
                 sliceNumber,
                 axisNumber,
                 lowerThreshold,
                 upperThreshold,
                 regionGrowingNode,  // This is the node for the image we are writing to.
                 regionGrowingImage  // This is the image we are writing to.
                )
            );

            // Copy to segmentation image.
            typedef itk::Image<unsigned char, 3> ImageType;
            typedef mitk::ImageToItk< ImageType > ImageToItkType;

            ImageToItkType::Pointer regionGrowingToItk = ImageToItkType::New();
            regionGrowingToItk->SetInput(regionGrowingImage);
            regionGrowingToItk->Update();

            ImageToItkType::Pointer outputToItk = ImageToItkType::New();
            outputToItk->SetInput(segmentationImage);
            outputToItk->Update();

            niftk::ITKCopyRegion<unsigned char, 3>(
                regionGrowingToItk->GetOutput(),
                axisNumber,
                sliceNumber,
                outputToItk->GetOutput()
                );

            regionGrowingToItk = NULL;
            outputToItk = NULL;

            // Update the current slice contours, to regenerate cleaned orange contours
            // around just the regions of interest that have a valid seed.
            this->UpdateCurrentSliceContours();
          }
          else
          {
            // Here we are not thresholding.

            // However, we can assume that all seeds are enclosed.
            // If the seeds were not all enclosed, the user received warning earlier,
            // and either abandoned this method, or accepted the warning and wiped the slice.

            AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
                niftk::ITKUpdateRegionGrowing, 3,
                (false,
                 *segmentationImage,
                 *seeds,
                 *segmentationContours,
                 *drawToolContours,
                 *polyToolContours,
                 sliceNumber,
                 axisNumber,
                 referenceImage->GetStatistics()->GetScalarValueMinNoRecompute(),
                 referenceImage->GetStatistics()->GetScalarValueMaxNoRecompute(),
                 regionGrowingNode,  // This is the node for the image we are writing to.
                 regionGrowingImage  // This is the image we are writing to.
                )
            );

          }

          // Then create filtered contours for the current slice.
          // So, if we are thresholding, we fit them round the current region growing image,
          // which if we have just used enclosed seeds above, will not include regions defined
          // by a seed and a threshold, but that have not been "applied" yet.

          AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
              niftk::ITKFilterContours, 3,
              (*segmentationImage,
               *seeds,
               *segmentationContours,
               *drawToolContours,
               *polyToolContours,
               axisNumber,
               sliceNumber,
               lowerThreshold,
               upperThreshold,
               m_GeneralSegmentorGUI->IsThresholdingCheckBoxChecked(),
               *(copyOfInputContourSet.GetPointer()),
               *(outputContourSet.GetPointer())
              )
            );

          mitk::UndoStackItem::IncCurrObjectEventId();
          mitk::UndoStackItem::IncCurrGroupEventId();
          mitk::UndoStackItem::ExecuteIncrement();

          niftk::OpClean *doOp = new niftk::OpClean(niftk::OP_CLEAN, true, outputContourSet);
          niftk::OpClean *undoOp = new niftk::OpClean(niftk::OP_CLEAN, false, copyOfInputContourSet);
          mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Clean: Filtering contours");
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
          this->ExecuteOperation(doOp);

          // Then we update the region growing to get up-to-date contours.
          this->UpdateRegionGrowing();

          if (!m_GeneralSegmentorGUI->IsThresholdingCheckBoxChecked())
          {
            // Then we "apply" this region growing.
            niftk::OpThresholdApply::ProcessorPointer processor = niftk::OpThresholdApply::ProcessorType::New();
            niftk::OpThresholdApply *doApplyOp = new niftk::OpThresholdApply(niftk::OP_THRESHOLD_APPLY, true, outputRegion, processor, m_GeneralSegmentorGUI->IsThresholdingCheckBoxChecked());
            niftk::OpThresholdApply *undoApplyOp = new niftk::OpThresholdApply(niftk::OP_THRESHOLD_APPLY, false, outputRegion, processor, m_GeneralSegmentorGUI->IsThresholdingCheckBoxChecked());
            mitk::OperationEvent* operationApplyEvent = new mitk::OperationEvent( m_Interface, doApplyOp, undoApplyOp, "Clean: Calculate new image");
            mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationApplyEvent );
            this->ExecuteOperation(doApplyOp);

            // We should update the current slice contours, as the green contours
            // are the current segmentation that will be applied when we change slice.
            this->UpdateCurrentSliceContours();
          }

          drawTool->Clean(sliceNumber, axisNumber);

          segmentationImage->Modified();
          segmentationNode->Modified();

          cleanWasPerformed = true;

        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Could not do clean command: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        }
        catch( itk::ExceptionObject &err )
        {
          MITK_ERROR << "Could not do clean command: Caught itk::ExceptionObject:" << err.what() << std::endl;
        }

        m_IsUpdating = false;

      }
      else
      {
        MITK_ERROR << "Could not do clean operation: Error axisNumber=" << axisNumber << ", sliceNumber=" << sliceNumber << std::endl;
      }
    }
  }

  if (cleanWasPerformed)
  {
    this->RequestRenderWindowUpdate();
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnWipeButtonClicked()
{
  this->DoWipe(0);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnWipePlusButtonClicked()
{

  MIDASOrientation midasOrientation = this->GetOrientationAsEnum();

  QString orientationText;
  QString messageWithOrientation = "All slices %1 the present will be cleared \nAre you sure?";

  if (midasOrientation == MIDAS_ORIENTATION_AXIAL)
  {
    orientationText = "superior to";
  }
  else if (midasOrientation == MIDAS_ORIENTATION_SAGITTAL)
  {
    orientationText = "right of";
  }
  else if (midasOrientation == MIDAS_ORIENTATION_CORONAL)
  {
    orientationText = "anterior to";
  }
  else
  {
    orientationText = "up from";
  }

  int returnValue = QMessageBox::warning(this->GetParent(), tr("NiftyView"),
                                                            tr(messageWithOrientation.toStdString().c_str()).arg(orientationText),
                                                            QMessageBox::Yes | QMessageBox::No);
  if (returnValue == QMessageBox::No)
  {
    return;
  }

  this->DoWipe(1);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnWipeMinusButtonClicked()
{
  MIDASOrientation midasOrientation = this->GetOrientationAsEnum();

  QString orientationText;
  QString messageWithOrientation = "All slices %1 the present will be cleared \nAre you sure?";

  if (midasOrientation == MIDAS_ORIENTATION_AXIAL)
  {
    orientationText = "inferior to";
  }
  else if (midasOrientation == MIDAS_ORIENTATION_SAGITTAL)
  {
    orientationText = "left of";
  }
  else if (midasOrientation == MIDAS_ORIENTATION_CORONAL)
  {
    orientationText = "posterior to";
  }
  else
  {
    orientationText = "down from";
  }

  int returnValue = QMessageBox::warning(this->GetParent(), tr("NiftyView"),
                                                            tr(messageWithOrientation.toStdString().c_str()).arg(orientationText),
                                                            QMessageBox::Yes | QMessageBox::No);
  if (returnValue == QMessageBox::No)
  {
    return;
  }

  this->DoWipe(-1);
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorView::DoWipe(int direction)
{
  bool wipeWasPerformed = false;

  if (!this->HasInitialisedWorkingData())
  {
    return wipeWasPerformed;
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {

    mitk::DataNode::Pointer segmentationNode = this->GetWorkingData()[niftk::MIDASTool::SEGMENTATION];
    mitk::Image::Pointer segmentationImage = this->GetWorkingImageFromToolManager(niftk::MIDASTool::SEGMENTATION);

    if (segmentationImage.IsNotNull() && segmentationNode.IsNotNull())
    {
      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
      int axisNumber = this->GetViewAxis();
      int upDirection = this->GetUpDirection();

      if (direction != 0) // zero means, current slice.
      {
        direction = direction*upDirection;
      }

      mitk::PointSet::Pointer copyOfInputSeeds = mitk::PointSet::New();
      mitk::PointSet::Pointer outputSeeds = mitk::PointSet::New();
      std::vector<int> outputRegion;

      if (axisNumber != -1 && sliceNumber != -1)
      {

        m_IsUpdating = true;

        try
        {

          mitk::ToolManager *toolManager = this->GetToolManager();
          assert(toolManager);

          niftk::MIDASDrawTool *drawTool = static_cast<niftk::MIDASDrawTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASDrawTool>()));
          assert(drawTool);

          if (toolManager->GetActiveToolID() == toolManager->GetToolIdByToolType<niftk::MIDASPolyTool>())
          {
            toolManager->ActivateTool(-1);
          }


          if (direction == 0)
          {
            mitk::CopyPointSets(*seeds, *copyOfInputSeeds);
            mitk::CopyPointSets(*seeds, *outputSeeds);

            AccessFixedDimensionByItk_n(segmentationImage,
                niftk::ITKCalculateSliceRegionAsVector, 3,
                (axisNumber,
                 sliceNumber,
                 outputRegion
                )
              );

          }
          else
          {
            AccessFixedDimensionByItk_n(segmentationImage, // The binary image = current segmentation
                niftk::ITKPreProcessingForWipe, 3,
                (*seeds,
                 sliceNumber,
                 axisNumber,
                 direction,
                 *(copyOfInputSeeds.GetPointer()),
                 *(outputSeeds.GetPointer()),
                 outputRegion
                )
              );
          }

          mitk::UndoStackItem::IncCurrObjectEventId();
          mitk::UndoStackItem::IncCurrGroupEventId();
          mitk::UndoStackItem::ExecuteIncrement();

          niftk::OpWipe::ProcessorPointer processor = niftk::OpWipe::ProcessorType::New();
          niftk::OpWipe *doOp = new niftk::OpWipe(niftk::OP_WIPE, true, sliceNumber, axisNumber, outputRegion, outputSeeds, processor);
          niftk::OpWipe *undoOp = new niftk::OpWipe(niftk::OP_WIPE, false, sliceNumber, axisNumber, outputRegion, copyOfInputSeeds, processor);
          mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Wipe command");
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
          this->ExecuteOperation(doOp);

          drawTool->ClearWorkingData();
          this->UpdateCurrentSliceContours();

          // Successful outcome.
          wipeWasPerformed = true;
        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Could not do wipe command: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        }
        catch( itk::ExceptionObject &err )
        {
          MITK_ERROR << "Could not do wipe command: Caught itk::ExceptionObject:" << err.what() << std::endl;
        }

        m_IsUpdating = false;
      }
      else
      {
        MITK_ERROR << "Could not wipe: Error, axisNumber=" << axisNumber << ", sliceNumber=" << sliceNumber << std::endl;
      }
    }
  }

  if (wipeWasPerformed)
  {
    this->RequestRenderWindowUpdate();
  }

  return wipeWasPerformed;
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnPropagate3DButtonClicked()
{
  this->DoPropagate(false, true);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnPropagateUpButtonClicked()
{
  this->DoPropagate(true, false);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnPropagateDownButtonClicked()
{
  this->DoPropagate(false, false);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::DoPropagate(bool isUp, bool is3D)
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  MIDASOrientation midasOrientation = this->GetOrientationAsEnum();
  itk::Orientation orientation = niftk::GetItkOrientation(midasOrientation);

  QString message;

  if (is3D)
  {
    message = "All slices will be over-written";
  }
  else
  {
    QString orientationText;
    QString messageWithOrientation = "All slices %1 the present will be over-written";

    if (isUp)
    {
      if (midasOrientation == MIDAS_ORIENTATION_AXIAL)
      {
        orientationText = "superior to";
      }
      else if (midasOrientation == MIDAS_ORIENTATION_SAGITTAL)
      {
        orientationText = "right of";
      }
      else if (midasOrientation == MIDAS_ORIENTATION_CORONAL)
      {
        orientationText = "anterior to";
      }
      else
      {
        orientationText = "up from";
      }
    }
    else if (!isUp)
    {
      if (midasOrientation == MIDAS_ORIENTATION_AXIAL)
      {
        orientationText = "inferior to";
      }
      else if (midasOrientation == MIDAS_ORIENTATION_SAGITTAL)
      {
        orientationText = "left of";
      }
      else if (midasOrientation == MIDAS_ORIENTATION_CORONAL)
      {
        orientationText = "posterior to";
      }
      else
      {
        orientationText = "up from";
      }
    }

    message = tr(messageWithOrientation.toStdString().c_str()).arg(orientationText);
  }

  int returnValue = QMessageBox::warning(this->GetParent(), tr("NiftyView"),
                                                   tr("%1.\n"
                                                      "Are you sure?").arg(message),
                                                   QMessageBox::Yes | QMessageBox::No);
  if (returnValue == QMessageBox::No)
  {
    return;
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {

    mitk::DataNode::Pointer segmentationNode = this->GetWorkingData()[niftk::MIDASTool::SEGMENTATION];
    mitk::Image::Pointer segmentationImage = this->GetWorkingImageFromToolManager(niftk::MIDASTool::SEGMENTATION);

    if (segmentationImage.IsNotNull() && segmentationNode.IsNotNull())
    {

      mitk::DataNode::Pointer regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(niftk::MIDASTool::REGION_GROWING_NAME.c_str(), segmentationNode, true);
      assert(regionGrowingNode);

      mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
      assert(regionGrowingImage);

      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      mitk::ToolManager *toolManager = this->GetToolManager();
      assert(toolManager);

      niftk::MIDASDrawTool *drawTool = static_cast<niftk::MIDASDrawTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASDrawTool>()));
      assert(drawTool);

      double lowerThreshold = m_GeneralSegmentorGUI->GetLowerThreshold();
      double upperThreshold = m_GeneralSegmentorGUI->GetUpperThreshold();
      int sliceNumber = this->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
      int axisNumber = this->GetViewAxis();
      int direction = this->GetUpDirection();
      if (!is3D && !isUp)
      {
        direction *= -1;
      }
      else if (is3D)
      {
        direction = 0;
      }

      mitk::PointSet::Pointer copyOfInputSeeds = mitk::PointSet::New();
      mitk::PointSet::Pointer outputSeeds = mitk::PointSet::New();
      std::vector<int> outputRegion;

      if (axisNumber != -1 && sliceNumber != -1 && orientation != itk::ORIENTATION_UNKNOWN)
      {

        m_IsUpdating = true;

        try
        {
          AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
              niftk::ITKPropagateToRegionGrowingImage, 3,
              (*seeds,
               sliceNumber,
               axisNumber,
               direction,
               lowerThreshold,
               upperThreshold,
               *(copyOfInputSeeds.GetPointer()),
               *(outputSeeds.GetPointer()),
               outputRegion,
               regionGrowingNode,  // This is the node for the image we are writing to.
               regionGrowingImage  // This is the image we are writing to.
              )
            );

          if (toolManager->GetActiveToolID() == toolManager->GetToolIdByToolType<niftk::MIDASPolyTool>())
          {
            toolManager->ActivateTool(-1);
          }

          mitk::UndoStackItem::IncCurrObjectEventId();
          mitk::UndoStackItem::IncCurrGroupEventId();
          mitk::UndoStackItem::ExecuteIncrement();

          QString message = tr("Propagate: copy region growing");
          niftk::OpPropagate::ProcessorPointer processor = niftk::OpPropagate::ProcessorType::New();
          niftk::OpPropagate *doPropOp = new niftk::OpPropagate(niftk::OP_PROPAGATE, true, outputRegion, processor);
          niftk::OpPropagate *undoPropOp = new niftk::OpPropagate(niftk::OP_PROPAGATE, false, outputRegion, processor);
          mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doPropOp, undoPropOp, message.toStdString());
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
          this->ExecuteOperation(doPropOp);

          message = tr("Propagate: copy seeds");
          niftk::OpPropagateSeeds *doPropSeedsOp = new niftk::OpPropagateSeeds(niftk::OP_PROPAGATE_SEEDS, true, sliceNumber, axisNumber, outputSeeds);
          niftk::OpPropagateSeeds *undoPropSeedsOp = new niftk::OpPropagateSeeds(niftk::OP_PROPAGATE_SEEDS, false, sliceNumber, axisNumber, copyOfInputSeeds);
          mitk::OperationEvent* operationPropEvent = new mitk::OperationEvent( m_Interface, doPropSeedsOp, undoPropSeedsOp, message.toStdString());
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationPropEvent );
          this->ExecuteOperation(doPropOp);

          drawTool->ClearWorkingData();
          this->UpdateCurrentSliceContours(false);
          this->UpdateRegionGrowing(false);
        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Could not propagate: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        }
        catch( itk::ExceptionObject &err )
        {
          MITK_ERROR << "Could not propagate: Caught itk::ExceptionObject:" << err.what() << std::endl;
        }

        m_IsUpdating = false;
      }
      else
      {
        MITK_ERROR << "Could not propagate: Error axisNumber=" << axisNumber << ", sliceNumber=" << sliceNumber << ", orientation=" << orientation << ", direction=" << direction << std::endl;
      }
    }
  }

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnAnyButtonClicked()
{
  /// Set the focus back to the main window. This is needed so that the keyboard shortcuts
  /// (like 'a' and 'z' for changing slice) keep on working.
  if (QmitkRenderWindow* mainWindow = this->GetSelectedRenderWindow())
  {
    mainWindow->setFocus();
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::NodeChanged(const mitk::DataNode* node)
{
  if (   m_IsDeleting
      || m_IsUpdating
      || !this->HasInitialisedWorkingData()
      )
  {
    return;
  }

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
  if (workingData.size() > 0)
  {
    bool seedsChanged(false);
    bool drawContoursChanged(false);

    if (workingData[niftk::MIDASTool::SEEDS] != NULL && workingData[niftk::MIDASTool::SEEDS] == node)
    {
      seedsChanged = true;
    }
    if (workingData[niftk::MIDASTool::DRAW_CONTOURS] != NULL && workingData[niftk::MIDASTool::DRAW_CONTOURS] == node)
    {
      drawContoursChanged = true;
    }

    if (!seedsChanged && !drawContoursChanged)
    {
      return;
    }

    mitk::DataNode::Pointer segmentationImageNode = workingData[niftk::MIDASTool::SEGMENTATION];
    if (segmentationImageNode.IsNotNull())
    {
      mitk::PointSet* seeds = this->GetSeeds();
      if (seeds != NULL && seeds->GetSize() > 0)
      {

        bool contourIsBeingEdited(false);
        if (segmentationImageNode.GetPointer() == node)
        {
          segmentationImageNode->GetBoolProperty(niftk::MIDASContourTool::EDITING_PROPERTY_NAME.c_str(), contourIsBeingEdited);
        }

        if (!contourIsBeingEdited)
        {
          if (seedsChanged)
          {
            this->RecalculateMinAndMaxOfSeedValues();
          }

          if (seedsChanged || drawContoursChanged)
          {
            this->UpdateRegionGrowing();
          }
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::NodeRemoved(const mitk::DataNode* removedNode)
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::DataNode::Pointer segmentationNode = this->GetToolManager()->GetWorkingData(niftk::MIDASTool::SEGMENTATION);

  if (segmentationNode.GetPointer() == removedNode)
  {
    this->DiscardSegmentation();
  }
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::OnContoursChanged()
{
  this->UpdateRegionGrowing();
}


/**************************************************************
 * End of: The main MIDAS business logic.
 *************************************************************/

/******************************************************************
 * Start of ExecuteOperation - main method in Undo/Redo framework.
 *
 * Notes: In this method, we update items, using the given
 * operation. We do not know if this is a "Undo" or a "Redo"
 * type of operation. We can set the modified field.
 * But do not be tempted to put things like:
 *
 * this->RequestRenderWindowUpdate();
 *
 * or
 *
 * this->UpdateRegionGrowing() etc.
 *
 * as these methods may be called multiple times during one user
 * operation. So the methods creating the mitk::Operation objects
 * should also be the ones deciding when we update the display.
 ******************************************************************/

void niftkGeneralSegmentorView::ExecuteOperation(mitk::Operation* operation)
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  if (!operation) return;

  mitk::Image::Pointer segmentationImage = this->GetWorkingImageFromToolManager(niftk::MIDASTool::SEGMENTATION);
  assert(segmentationImage);

  mitk::DataNode::Pointer segmentationNode = this->GetWorkingData()[niftk::MIDASTool::SEGMENTATION];
  assert(segmentationNode);

  mitk::Image* referenceImage = this->GetReferenceImageFromToolManager();
  assert(referenceImage);

  mitk::Image* regionGrowingImage = this->GetWorkingImageFromToolManager(niftk::MIDASTool::REGION_GROWING);
  assert(regionGrowingImage);

  mitk::PointSet* seeds = this->GetSeeds();
  assert(seeds);

  mitk::DataNode::Pointer seedsNode = this->GetWorkingData()[niftk::MIDASTool::SEEDS];
  assert(seedsNode);

  mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();
  assert(renderWindowPart);

  switch (operation->GetOperationType())
  {
  case niftk::OP_CHANGE_SLICE:
    {
      // Simply to make sure we can switch slice, and undo/redo it.
      niftk::OpChangeSliceCommand* op = dynamic_cast<niftk::OpChangeSliceCommand*>(operation);
      assert(op);

      mitk::Point3D currentPoint = renderWindowPart->GetSelectedPosition();

      mitk::Point3D beforePoint = op->GetBeforePoint();
      mitk::Point3D afterPoint = op->GetAfterPoint();
      int beforeSlice = op->GetBeforeSlice();
      int afterSlice = op->GetAfterSlice();

      mitk::Point3D selectedPoint;

      if (op->IsRedo())
      {
        selectedPoint = afterPoint;
      }
      else
      {
        selectedPoint = beforePoint;
      }

      // Only move if we are not already on this slice.
      // Better to compare integers than floating point numbers.
      if (beforeSlice != afterSlice)
      {
        m_IsChangingSlice = true;
        renderWindowPart->SetSelectedPosition(selectedPoint);
        m_IsChangingSlice = false;
      }

      break;
    }
  case niftk::OP_PROPAGATE_SEEDS:
    {
      niftk::OpPropagateSeeds* op = dynamic_cast<niftk::OpPropagateSeeds*>(operation);
      assert(op);

      mitk::PointSet* newSeeds = op->GetSeeds();
      assert(newSeeds);

      mitk::CopyPointSets(*newSeeds, *seeds);

      seeds->Modified();
      seedsNode->Modified();

      break;
    }
  case niftk::OP_RETAIN_MARKS:
    {
      try
      {
        niftk::OpRetainMarks* op = static_cast<niftk::OpRetainMarks*>(operation);
        assert(op);

        niftk::OpRetainMarks::ProcessorType::Pointer processor = op->GetProcessor();
        bool redo = op->IsRedo();
        int fromSlice = op->GetFromSlice();
        int toSlice = op->GetToSlice();
        itk::Orientation orientation = op->GetOrientation();

        typedef itk::Image<mitk::Tool::DefaultSegmentationDataType, 3> BinaryImage3DType;
        typedef mitk::ImageToItk< BinaryImage3DType > SegmentationImageToItkType;
        SegmentationImageToItkType::Pointer targetImageToItk = SegmentationImageToItkType::New();
        targetImageToItk->SetInput(segmentationImage);
        targetImageToItk->Update();

        processor->SetSourceImage(targetImageToItk->GetOutput());
        processor->SetDestinationImage(targetImageToItk->GetOutput());
        processor->SetSlices(orientation, fromSlice, toSlice);

        if (redo)
        {
          processor->Redo();
        }
        else
        {
          processor->Undo();
        }

        targetImageToItk = NULL;

        mitk::Image::Pointer outputImage = mitk::ImportItkImage( processor->GetDestinationImage());

        processor->SetSourceImage(NULL);
        processor->SetDestinationImage(NULL);

        segmentationNode->SetData(outputImage);
        segmentationNode->Modified();
      }
      catch( itk::ExceptionObject &err )
      {
        MITK_ERROR << "Could not do retain marks: Caught itk::ExceptionObject:" << err.what() << std::endl;
        return;
      }

      break;
    }
  case niftk::OP_THRESHOLD_APPLY:
    {
      niftk::OpThresholdApply *op = dynamic_cast<niftk::OpThresholdApply*>(operation);
      assert(op);

      try
      {
        AccessFixedDimensionByItk_n(referenceImage, niftk::ITKPropagateToSegmentationImage, 3,
              (
                segmentationImage,
                regionGrowingImage,
                op
              )
            );

        m_GeneralSegmentorGUI->SetThresholdingCheckBoxChecked(op->GetThresholdFlag());
        m_GeneralSegmentorGUI->SetThresholdingWidgetsEnabled(op->GetThresholdFlag());

        segmentationImage->Modified();
        segmentationNode->Modified();

        regionGrowingImage->Modified();

      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Could not do threshold: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        return;
      }
      catch( itk::ExceptionObject &err )
      {
        MITK_ERROR << "Could not do threshold: Caught itk::ExceptionObject:" << err.what() << std::endl;
        return;
      }

      break;
    }
  case niftk::OP_CLEAN:
    {
      try
      {
        niftk::OpClean* op = dynamic_cast<niftk::OpClean*>(operation);
        assert(op);

        mitk::ContourModelSet* newContours = op->GetContourSet();
        assert(newContours);

        mitk::ContourModelSet* contoursToReplace = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[niftk::MIDASTool::CONTOURS]->GetData());
        assert(contoursToReplace);

        niftk::MIDASContourTool::CopyContourSet(*newContours, *contoursToReplace);
        contoursToReplace->Modified();
        this->GetWorkingData()[niftk::MIDASTool::CONTOURS]->Modified();

        segmentationImage->Modified();
        segmentationNode->Modified();

      }
      catch( itk::ExceptionObject &err )
      {
        MITK_ERROR << "Could not do clean: Caught itk::ExceptionObject:" << err.what() << std::endl;
        return;
      }

      break;
    }
  case niftk::OP_WIPE:
    {
      niftk::OpWipe *op = dynamic_cast<niftk::OpWipe*>(operation);
      assert(op);

      try
      {
        AccessFixedTypeByItk_n(segmentationImage,
            niftk::ITKDoWipe,
            (unsigned char),
            (3),
              (
                seeds,
                op
              )
            );

        segmentationImage->Modified();
        segmentationNode->Modified();

      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Could not do wipe: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        return;
      }
      catch( itk::ExceptionObject &err )
      {
        MITK_ERROR << "Could not do wipe: Caught itk::ExceptionObject:" << err.what() << std::endl;
        return;
      }

      break;
    }
  case niftk::OP_PROPAGATE:
    {
      niftk::OpPropagate *op = dynamic_cast<niftk::OpPropagate*>(operation);
      assert(op);

      try
      {
        AccessFixedDimensionByItk_n(referenceImage, niftk::ITKPropagateToSegmentationImage, 3,
              (
                segmentationImage,
                regionGrowingImage,
                op
              )
            );

        segmentationImage->Modified();
        segmentationNode->Modified();

      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Could not do propagation: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        return;
      }
      catch( itk::ExceptionObject &err )
      {
        MITK_ERROR << "Could not do propagation: Caught itk::ExceptionObject:" << err.what() << std::endl;
        return;
      }
      break;
    }
  default:;
  }
}

/******************************************************************
 * End of ExecuteOperation - main method in Undo/Redo framework.
 ******************************************************************/
