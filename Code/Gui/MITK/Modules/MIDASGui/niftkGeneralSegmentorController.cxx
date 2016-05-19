/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorController.h"

#include <QMessageBox>

#include <mitkImageAccessByItk.h>
#include <mitkImageStatisticsHolder.h>
#include <mitkITKImageImport.h>
#include <mitkOperationEvent.h>
#include <mitkPointSet.h>
#include <mitkUndoController.h>

#include <QmitkRenderWindow.h>

#include <mitkDataStorageUtils.h>

#include <niftkIBaseView.h>
#include <niftkGeneralSegmentorUtils.h>
#include <niftkMIDASDrawTool.h>
#include <niftkMIDASSeedTool.h>
#include <niftkMIDASPolyTool.h>
#include <niftkMIDASPosnTool.h>

#include "Internal/niftkGeneralSegmentorGUI.h"

namespace niftk
{

class GeneralSegmentorControllerPrivate
{
  Q_DECLARE_PUBLIC(GeneralSegmentorController);

  GeneralSegmentorController* const q_ptr;

public:

  GeneralSegmentorControllerPrivate(GeneralSegmentorController* q);
  ~GeneralSegmentorControllerPrivate();

  /// \brief All the GUI controls for the main view part.
  GeneralSegmentorGUI* m_GUI;

  /// \brief Pointer to interface object, used as callback in Undo/Redo framework
  GeneralSegmentorEventInterface::Pointer m_Interface;

  /// \brief This class hooks into the Global Interaction system to respond to Key press events.
  MIDASToolKeyPressStateMachine::Pointer m_ToolKeyPressStateMachine;

  /// \brief Selected orientation in the viewer.
  ImageOrientation m_Orientation;

  /// \brief Index of the selected slice in world space.
  int m_SelectedSliceIndex;

  /// \brief Keeps track of the previous slice index and reset to -1 when the window focus changes.
  /// The slice index is in terms of the reference image coordinates (voxel space), not the coordinates
  /// of the renderer (world space).
  int m_SliceIndex;

  /// \brief We track the previous selected position, as it is used in calculations of which slice we are on,
  /// as under certain conditions, you can't just take the slice index from the slice navigation controller.
  mitk::Point3D m_SelectedPosition;

  /// \brief Flag to stop re-entering code, while updating.
  bool m_IsUpdating;

  /// \brief Flag to stop re-entering code, while trying to delete/clear the pipeline.
  bool m_IsDeleting;

  /// \brief Additional flag to stop re-entering code, specifically to block
  /// slice change commands from the slice navigation controller.
  bool m_IsChangingSlice;

  bool m_IsRestarting;
};

//-----------------------------------------------------------------------------
GeneralSegmentorControllerPrivate::GeneralSegmentorControllerPrivate(GeneralSegmentorController* generalSegmentorController)
  : q_ptr(generalSegmentorController),
    m_ToolKeyPressStateMachine(nullptr),
    m_Orientation(IMAGE_ORIENTATION_UNKNOWN),
    m_SelectedSliceIndex(-1),
    m_SliceIndex(-1),
    m_IsUpdating(false),
    m_IsDeleting(false),
    m_IsChangingSlice(false),
    m_IsRestarting(false)
{
  Q_Q(GeneralSegmentorController);

  m_Interface = GeneralSegmentorEventInterface::New();
  m_Interface->SetGeneralSegmentorController(q);

  mitk::ToolManager* toolManager = q->GetToolManager();
  toolManager->RegisterTool("MIDASDrawTool");
  toolManager->RegisterTool("MIDASSeedTool");
  toolManager->RegisterTool("MIDASPolyTool");
  toolManager->RegisterTool("MIDASPosnTool");

  q->GetToolByType<MIDASDrawTool>()->InstallEventFilter(q);
  q->GetToolByType<MIDASSeedTool>()->InstallEventFilter(q);
  q->GetToolByType<MIDASPolyTool>()->InstallEventFilter(q);
  q->GetToolByType<MIDASPosnTool>()->InstallEventFilter(q);

//  m_ToolKeyPressStateMachine = MIDASToolKeyPressStateMachine::New("MIDASToolKeyPressStateMachine", q);
  m_ToolKeyPressStateMachine = MIDASToolKeyPressStateMachine::New(q);

  m_SelectedPosition.Fill(0);
}


//-----------------------------------------------------------------------------
GeneralSegmentorControllerPrivate::~GeneralSegmentorControllerPrivate()
{
}


//-----------------------------------------------------------------------------
GeneralSegmentorController::GeneralSegmentorController(IBaseView* view)
  : BaseSegmentorController(view),
    d_ptr(new GeneralSegmentorControllerPrivate(this))
{
}


//-----------------------------------------------------------------------------
GeneralSegmentorController::~GeneralSegmentorController()
{
  this->GetToolByType<MIDASDrawTool>()->RemoveEventFilter(this);
  this->GetToolByType<MIDASSeedTool>()->RemoveEventFilter(this);
  this->GetToolByType<MIDASPolyTool>()->RemoveEventFilter(this);
  this->GetToolByType<MIDASPosnTool>()->RemoveEventFilter(this);
}


//-----------------------------------------------------------------------------
BaseGUI* GeneralSegmentorController::CreateGUI(QWidget* parent)
{
  return new GeneralSegmentorGUI(parent);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::SetupGUI(QWidget* parent)
{
  Q_D(GeneralSegmentorController);

  BaseSegmentorController::SetupGUI(parent);

  d->m_GUI = dynamic_cast<GeneralSegmentorGUI*>(this->GetSegmentorGUI());

  this->connect(d->m_GUI, SIGNAL(CleanButtonClicked()), SLOT(OnCleanButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(WipeButtonClicked()), SLOT(OnWipeButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(WipePlusButtonClicked()), SLOT(OnWipePlusButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(WipeMinusButtonClicked()), SLOT(OnWipeMinusButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(PropagateUpButtonClicked()), SLOT(OnPropagateUpButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(PropagateDownButtonClicked()), SLOT(OnPropagateDownButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(Propagate3DButtonClicked()), SLOT(OnPropagate3DButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(OKButtonClicked()), SLOT(OnOKButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(CancelButtonClicked()), SLOT(OnCancelButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(RestartButtonClicked()), SLOT(OnRestartButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(ResetButtonClicked()), SLOT(OnResetButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(ThresholdApplyButtonClicked()), SLOT(OnThresholdApplyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(ThresholdingCheckBoxToggled(bool)), SLOT(OnThresholdingCheckBoxToggled(bool)));
  this->connect(d->m_GUI, SIGNAL(SeePriorCheckBoxToggled(bool)), SLOT(OnSeePriorCheckBoxToggled(bool)));
  this->connect(d->m_GUI, SIGNAL(SeeNextCheckBoxToggled(bool)), SLOT(OnSeeNextCheckBoxToggled(bool)));
  this->connect(d->m_GUI, SIGNAL(ThresholdValueChanged()), SLOT(OnThresholdValueChanged()));

  /// Transfer the focus back to the main window if any button is pressed.
  /// This is needed so that the key interactions (like 'a'/'z' for changing slice) keep working.
  this->connect(d->m_GUI, SIGNAL(NewSegmentationButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(CleanButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(WipeButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(WipePlusButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(WipeMinusButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(PropagateUpButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(PropagateDownButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(Propagate3DButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(OKButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(CancelButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(RestartButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(ResetButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(ThresholdApplyButtonClicked()), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(ThresholdingCheckBoxToggled(bool)), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(SeePriorCheckBoxToggled(bool)), SLOT(OnAnyButtonClicked()));
  this->connect(d->m_GUI, SIGNAL(SeeNextCheckBoxToggled(bool)), SLOT(OnAnyButtonClicked()));
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::IsASegmentationImage(const mitk::DataNode::Pointer node)
{
  assert(node);
  bool result = false;

  if (mitk::IsNodeABinaryImage(node))
  {

    mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDataStorage(), node, false);

    if (parent.IsNotNull())
    {
      mitk::DataStorage* dataStorage = this->GetDataStorage();
      mitk::DataNode::Pointer seedsNode = dataStorage->GetNamedDerivedNode(MIDASTool::SEEDS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer currentContoursNode = dataStorage->GetNamedDerivedNode(MIDASTool::CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer drawContoursNode = dataStorage->GetNamedDerivedNode(MIDASTool::DRAW_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer seePriorContoursNode = dataStorage->GetNamedDerivedNode(MIDASTool::PRIOR_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer seeNextContoursNode = dataStorage->GetNamedDerivedNode(MIDASTool::NEXT_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer regionGrowingImageNode = dataStorage->GetNamedDerivedNode(MIDASTool::REGION_GROWING_NAME.c_str(), node, true);

      if (seedsNode.IsNotNull()
          && currentContoursNode.IsNotNull()
          && drawContoursNode.IsNotNull()
          && seePriorContoursNode.IsNotNull()
          && seeNextContoursNode.IsNotNull()
          && regionGrowingImageNode.IsNotNull()
          )
      {
        result = true;
      }
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType GeneralSegmentorController::GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  assert(node);
  mitk::ToolManager::DataVectorType result;

  if (mitk::IsNodeABinaryImage(node))
  {
    mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDataStorage(), node, false);

    if (parent.IsNotNull())
    {
      mitk::DataStorage* dataStorage = this->GetDataStorage();
      mitk::DataNode::Pointer seedsNode = dataStorage->GetNamedDerivedNode(MIDASTool::SEEDS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer currentContoursNode = dataStorage->GetNamedDerivedNode(MIDASTool::CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer drawContoursNode = dataStorage->GetNamedDerivedNode(MIDASTool::DRAW_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer seePriorContoursNode = dataStorage->GetNamedDerivedNode(MIDASTool::PRIOR_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer seeNextContoursNode = dataStorage->GetNamedDerivedNode(MIDASTool::NEXT_CONTOURS_NAME.c_str(), node, true);
      mitk::DataNode::Pointer regionGrowingImageNode = dataStorage->GetNamedDerivedNode(MIDASTool::REGION_GROWING_NAME.c_str(), node, true);
      mitk::DataNode::Pointer initialSegmentationImageNode = dataStorage->GetNamedDerivedNode(MIDASTool::INITIAL_SEGMENTATION_NAME.c_str(), node, true);
      mitk::DataNode::Pointer initialSeedsNode = dataStorage->GetNamedDerivedNode(MIDASTool::INITIAL_SEEDS_NAME.c_str(), node, true);

      if (seedsNode.IsNotNull()
          && currentContoursNode.IsNotNull()
          && drawContoursNode.IsNotNull()
          && seePriorContoursNode.IsNotNull()
          && seeNextContoursNode.IsNotNull()
          && regionGrowingImageNode.IsNotNull()
          && initialSegmentationImageNode.IsNotNull()
          && initialSeedsNode.IsNotNull()
          )
      {
        // The order of this list must match the order they were created in.
        result.push_back(node);
        result.push_back(seedsNode);
        result.push_back(currentContoursNode);
        result.push_back(drawContoursNode);
        result.push_back(seePriorContoursNode);
        result.push_back(seeNextContoursNode);
        result.push_back(regionGrowingImageNode);
        result.push_back(initialSegmentationImageNode);
        result.push_back(initialSeedsNode);
      }
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node)
{
  bool canRestart = false;

  if (node.IsNotNull() && mitk::IsNodeABinaryImage(node))
  {
    mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDataStorage(), node, false);
    if (parent.IsNotNull())
    {
      if (mitk::IsNodeAGreyScaleImage(parent))
      {
        canRestart = true;
      }
    }
  }

  return canRestart;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnNewSegmentationButtonClicked()
{
  Q_D(GeneralSegmentorController);

  BaseSegmentorController::OnNewSegmentationButtonClicked();

  // Create the new segmentation, either using a previously selected one, or create a new volume.
  mitk::DataNode::Pointer newSegmentation = nullptr;
  bool isRestarting = false;

  // Make sure we have a reference images... which should always be true at this point.
  mitk::Image* image = this->GetReferenceImage();
  if (image)
  {
    mitk::ToolManager::Pointer toolManager = this->GetToolManager();
    assert(toolManager);

    mitk::DataNode::Pointer selectedNode = this->GetSelectedNode();

    if (mitk::IsNodeABinaryImage(selectedNode)
        && this->CanStartSegmentationForBinaryNode(selectedNode)
        && !this->IsASegmentationImage(selectedNode)
        )
    {
      newSegmentation =  selectedNode;
      isRestarting = true;
    }
    else
    {
      newSegmentation = this->CreateNewSegmentation();

      // The above method returns nullptr if the user exited the colour selection dialog box.
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
    newSegmentation->SetBoolProperty(MIDASContourTool::EDITING_PROPERTY_NAME.c_str(), false);

    // Make sure these are up to date, even though we don't use them right now.
    image->GetStatistics()->GetScalarValueMin();
    image->GetStatistics()->GetScalarValueMax();

    // This creates the point set for the seeds.
    mitk::PointSet::Pointer pointSet = mitk::PointSet::New();
    mitk::DataNode::Pointer pointSetNode = mitk::DataNode::New();
    pointSetNode->SetData(pointSet);
    pointSetNode->SetProperty("name", mitk::StringProperty::New(MIDASTool::SEEDS_NAME));
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
    mitk::DataNode::Pointer currentContours = this->CreateContourSet(newSegmentation, 0,1,0, MIDASTool::CONTOURS_NAME, true, 97);
    mitk::DataNode::Pointer drawContours = this->CreateContourSet(newSegmentation, 0,1,0, MIDASTool::DRAW_CONTOURS_NAME, true, 98);
    mitk::DataNode::Pointer seeNextNode = this->CreateContourSet(newSegmentation, 0,1,1, MIDASTool::NEXT_CONTOURS_NAME, false, 95);
    mitk::DataNode::Pointer seePriorNode = this->CreateContourSet(newSegmentation, 0.68,0.85,0.90, MIDASTool::PRIOR_CONTOURS_NAME, false, 96);

    // Create the region growing image.
    mitk::DataNode::Pointer regionGrowingImageNode = this->CreateHelperImage(image, newSegmentation, 0,0,1, MIDASTool::REGION_GROWING_NAME, false, 94);

    // Create nodes to store the original segmentation and seeds, so that it can be restored if the Restart button is pressed.
    mitk::DataNode::Pointer initialSegmentationNode = mitk::DataNode::New();
    initialSegmentationNode->SetProperty("name", mitk::StringProperty::New(MIDASTool::INITIAL_SEGMENTATION_NAME));
    initialSegmentationNode->SetBoolProperty("helper object", true);
    initialSegmentationNode->SetBoolProperty("visible", false);
    initialSegmentationNode->SetProperty("layer", mitk::IntProperty::New(99));
    initialSegmentationNode->SetFloatProperty("opacity", 1.0f);
    initialSegmentationNode->SetColor(tmpColor);
    initialSegmentationNode->SetProperty("binaryimage.selectedcolor", tmpColorProperty);

    mitk::DataNode::Pointer initialSeedsNode = mitk::DataNode::New();
    initialSeedsNode->SetProperty("name", mitk::StringProperty::New(MIDASTool::INITIAL_SEEDS_NAME));
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
//    mitk::IRenderingManager* renderingManager = 0;
//    mitk::IRenderWindowPart* renderWindowPart = this->GetView()->GetActiveRenderWindowPart();
//    if (renderWindowPart)
//    {
//      renderingManager = renderWindowPart->GetRenderingManager();
//    }
//    if (renderingManager)
//    {
//      // Make sure these points and contours are not rendered in 3D, as there can be many of them if you "propagate",
//      // and furthermore, there seem to be several seg faults rendering contour code in 3D. Haven't investigated yet.
//      QList<vtkRenderWindow*> renderWindows = renderingManager->GetAllRegisteredVtkRenderWindows();
//      for (QList<vtkRenderWindow*>::const_iterator iter = renderWindows.begin(); iter != renderWindows.end(); ++iter)
//      {
//        if ( mitk::BaseRenderer::GetInstance((*iter))->GetMapperID() == mitk::BaseRenderer::Standard3D )
//        {
//          pointSetNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
//          seePriorNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
//          seeNextNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
//          currentContours->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
//          drawContours->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
//          initialSegmentationNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
//          initialSeedsNode->SetBoolProperty("visible", false, mitk::BaseRenderer::GetInstance((*iter)));
//        }
//      }
//    }

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
    workingData[MIDASTool::SEGMENTATION] = newSegmentation;
    workingData[MIDASTool::SEEDS] = pointSetNode;
    workingData[MIDASTool::CONTOURS] = currentContours;
    workingData[MIDASTool::DRAW_CONTOURS] = drawContours;
    workingData[MIDASTool::PRIOR_CONTOURS] = seePriorNode;
    workingData[MIDASTool::NEXT_CONTOURS] = seeNextNode;
    workingData[MIDASTool::REGION_GROWING] = regionGrowingImageNode;
    workingData[MIDASTool::INITIAL_SEGMENTATION] = initialSegmentationNode;
    workingData[MIDASTool::INITIAL_SEEDS] = initialSeedsNode;
    toolManager->SetWorkingData(workingData);

    if (isRestarting)
    {
      int sliceAxis = this->GetReferenceImageSliceAxis();
      int sliceIndex = this->GetReferenceImageSliceIndex();
      this->InitialiseSeedsForSlice(sliceAxis, sliceIndex);
      this->UpdateCurrentSliceContours();
    }

    this->StoreInitialSegmentation();

    // Setup GUI.
    d->m_GUI->SetAllWidgetsEnabled(true);
    d->m_GUI->SetThresholdingWidgetsEnabled(false);
    d->m_GUI->SetThresholdingCheckBoxEnabled(true);
    d->m_GUI->SetThresholdingCheckBoxChecked(false);

    this->GetView()->FocusOnCurrentWindow();

    this->UpdateCurrentSliceContours(false);
    this->UpdateRegionGrowing(false);
    this->RequestRenderWindowUpdate();
    d->m_SliceIndex = this->GetReferenceImageSliceIndex();
    d->m_SelectedPosition = this->GetSelectedPosition();

    this->WaitCursorOff();

  } // end if we have a reference image

  d->m_IsRestarting = isRestarting;

  // Finally, select the new segmentation node.
  this->GetView()->SetCurrentSelection(newSegmentation);
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::HasInitialisedWorkingData()
{
  bool result = false;

  mitk::ToolManager::DataVectorType nodes = this->GetWorkingData();
  if (nodes.size() > 0)
  {
    result = true;
  }

  return result;
}


/**************************************************************
 * Start of: Functions to create reference data (hidden nodes)
 *************************************************************/

//-----------------------------------------------------------------------------
mitk::DataNode::Pointer GeneralSegmentorController::CreateHelperImage(mitk::Image::Pointer referenceImage, mitk::DataNode::Pointer segmentationNode, float r, float g, float b, std::string name, bool visible, int layer)
{
  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::Tool* drawTool = this->GetToolByType<MIDASDrawTool>();
  assert(drawTool);

  mitk::ColorProperty::Pointer col = mitk::ColorProperty::New(r, g, b);

  mitk::DataNode::Pointer helperImageNode = drawTool->CreateEmptySegmentationNode( referenceImage, name, col->GetColor());
  helperImageNode->SetColor(col->GetColor());
  helperImageNode->SetProperty("binaryimage.selectedcolor", col);
  helperImageNode->SetBoolProperty("helper object", true);
  helperImageNode->SetBoolProperty("visible", visible);
  helperImageNode->SetProperty("layer", mitk::IntProperty::New(layer));

  this->ApplyDisplayOptions(helperImageNode);

  return helperImageNode;
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer GeneralSegmentorController::CreateContourSet(mitk::DataNode::Pointer segmentationNode, float r, float g, float b, std::string name, bool visible, int layer)
{
  mitk::ContourModelSet::Pointer contourSet = mitk::ContourModelSet::New();

  mitk::DataNode::Pointer contourSetNode = mitk::DataNode::New();

  contourSetNode->SetProperty("color", mitk::ColorProperty::New(r, g, b));
  contourSetNode->SetProperty("contour.color", mitk::ColorProperty::New(r, g, b));
  contourSetNode->SetFloatProperty("opacity", 1.0f);
  contourSetNode->SetProperty("name", mitk::StringProperty::New(name));
  contourSetNode->SetBoolProperty("helper object", true);
  contourSetNode->SetBoolProperty("visible", visible);
  contourSetNode->SetProperty("layer", mitk::IntProperty::New(layer));
  contourSetNode->SetData(contourSet);

  return contourSetNode;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::StoreInitialSegmentation()
{
  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::ToolManager::DataVectorType workingData = toolManager->GetWorkingData();

  mitk::DataNode* segmentationNode = workingData[MIDASTool::SEGMENTATION];
  mitk::DataNode* seedsNode = workingData[MIDASTool::SEEDS];
  mitk::DataNode* initialSegmentationNode = workingData[MIDASTool::INITIAL_SEGMENTATION];
  mitk::DataNode* initialSeedsNode = workingData[MIDASTool::INITIAL_SEEDS];

  initialSegmentationNode->SetData(dynamic_cast<mitk::Image*>(segmentationNode->GetData())->Clone());
  initialSeedsNode->SetData(dynamic_cast<mitk::PointSet*>(seedsNode->GetData())->Clone());
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnNodeVisibilityChanged(const mitk::DataNode* node, const mitk::BaseRenderer* /*renderer*/)
{
  Q_D(GeneralSegmentorController);

  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  std::vector<mitk::DataNode*> workingData = this->GetWorkingData();
  if (!workingData.empty() && node == workingData[MIDASTool::SEGMENTATION])
  {
    bool segmentationNodeVisibility;
    if (node->GetVisibility(segmentationNodeVisibility, 0) && segmentationNodeVisibility)
    {
      workingData[MIDASTool::SEEDS]->SetVisibility(true);
      workingData[MIDASTool::CONTOURS]->SetVisibility(true);
      workingData[MIDASTool::DRAW_CONTOURS]->SetVisibility(true);
      if (d->m_GUI->IsSeePriorCheckBoxChecked())
      {
        workingData[MIDASTool::PRIOR_CONTOURS]->SetVisibility(true);
      }
      if (d->m_GUI->IsSeeNextCheckBoxChecked())
      {
        workingData[MIDASTool::NEXT_CONTOURS]->SetVisibility(true);
      }
      if (d->m_GUI->IsThresholdingCheckBoxChecked())
      {
        workingData[MIDASTool::REGION_GROWING]->SetVisibility(true);
      }
      workingData[MIDASTool::INITIAL_SEGMENTATION]->SetVisibility(false);
      workingData[MIDASTool::INITIAL_SEEDS]->SetVisibility(false);

      mitk::ToolManager::Pointer toolManager = this->GetToolManager();
      MIDASPolyTool* polyTool = this->GetToolByType<MIDASPolyTool>();
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


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnViewGetsVisible()
{
  /// TODO
//  mitk::GlobalInteraction::GetInstance()->AddListener(d->m_ToolKeyPressStateMachine);

  // Connect registered tools back to here, so we can do seed processing logic here.
  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  MIDASPolyTool* midasPolyTool = this->GetToolByType<MIDASPolyTool>();
  midasPolyTool->ContoursHaveChanged += mitk::MessageDelegate<GeneralSegmentorController>(this, &GeneralSegmentorController::OnContoursChanged);

  MIDASDrawTool* midasDrawTool = this->GetToolByType<MIDASDrawTool>();
  midasDrawTool->ContoursHaveChanged += mitk::MessageDelegate<GeneralSegmentorController>(this, &GeneralSegmentorController::OnContoursChanged);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnViewGetsHidden()
{
  BaseSegmentorController::OnViewGetsHidden();

  /// TODO
//  mitk::GlobalInteraction::GetInstance()->RemoveListener(d->m_ToolKeyPressStateMachine);

  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  MIDASPolyTool* polyTool = this->GetToolByType<MIDASPolyTool>();
  polyTool->ContoursHaveChanged -= mitk::MessageDelegate<GeneralSegmentorController>(this, &GeneralSegmentorController::OnContoursChanged);

  MIDASDrawTool* drawTool = this->GetToolByType<MIDASDrawTool>();
  drawTool->ContoursHaveChanged -= mitk::MessageDelegate<GeneralSegmentorController>(this, &GeneralSegmentorController::OnContoursChanged);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnSelectedSliceChanged(ImageOrientation orientation, int selectedSliceIndex)
{
  Q_D(GeneralSegmentorController);

  if (orientation != d->m_Orientation || selectedSliceIndex != d->m_SelectedSliceIndex)
  {
    if (this->HasInitialisedWorkingData()
        && orientation != IMAGE_ORIENTATION_UNKNOWN)
    {
      int sliceAxis = this->GetReferenceImageSliceAxis();
      int sliceIndex = this->GetReferenceImageSliceIndex();
      mitk::Point3D selectedPosition = this->GetSelectedPosition();

      assert(sliceAxis >= 0);
      assert(sliceIndex >= 0);

      if (!d->m_IsUpdating
          && !d->m_IsChangingSlice)
      {
        mitk::Image* referenceImage = this->GetReferenceImage();
        mitk::Image* segmentationImage = this->GetWorkingImage(MIDASTool::SEGMENTATION);
        assert(referenceImage && segmentationImage);

        bool isThresholdingOn = d->m_GUI->IsThresholdingCheckBoxChecked();

        mitk::Operation* doOp;
        mitk::Operation* undoOp;

        /// Changing to previous or next slice of the same orientation.
        if (orientation == d->m_Orientation && std::abs(d->m_SliceIndex - sliceIndex) == 1)
        {
          itk::Orientation itkOrientation = GetItkOrientation(this->GetOrientation());

          mitk::ToolManager* toolManager = this->GetToolManager();
          MIDASDrawTool* drawTool = this->GetToolByType<MIDASDrawTool>();

          std::vector<int> outputRegion;
          mitk::PointSet::Pointer copyOfCurrentSeeds = mitk::PointSet::New();
          mitk::PointSet::Pointer propagatedSeeds = mitk::PointSet::New();
          mitk::PointSet* seeds = this->GetSeeds();
          bool previousSliceIsEmpty = false;
          bool sliceIsEmpty = true;
          bool operationCancelled = false;

          bool wasUpdating = d->m_IsUpdating;
          d->m_IsUpdating = true;

          try
          {
            ///////////////////////////////////////////////////////
            // See: https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK/issues/1742
            //      for the whole logic surrounding changing slice.
            ///////////////////////////////////////////////////////

            AccessFixedDimensionByItk_n(segmentationImage,
                ITKSliceIsEmpty, 3,
                (sliceAxis,
                 sliceIndex,
                 sliceIsEmpty
                )
              );

            if (d->m_GUI->IsRetainMarksCheckBoxChecked())
            {
              int answer = QMessageBox::NoButton;

              if (!isThresholdingOn)
              {
                AccessFixedDimensionByItk_n(segmentationImage,
                    ITKSliceIsEmpty, 3,
                    (sliceAxis,
                     d->m_SliceIndex,
                     previousSliceIsEmpty
                    )
                  );
              }

              if (previousSliceIsEmpty)
              {
                answer = QMessageBox::warning(d->m_GUI->GetParent(), tr("NiftyMIDAS"),
                                                        tr("The previous slice is empty - retain marks cannot be performed.\n"
                                                           "Use the 'wipe' functionality to erase slices instead"),
                                                        QMessageBox::Ok
                                     );
              }
              else if (!sliceIsEmpty)
              {
                answer = QMessageBox::warning(d->m_GUI->GetParent(), tr("NiftyMIDAS"),
                                                        tr("The new slice is not empty - retain marks will overwrite the slice.\n"
                                                           "Are you sure?"),
                                                        QMessageBox::Yes | QMessageBox::No);
              }

              if (answer == QMessageBox::Ok || answer == QMessageBox::No )
              {
                operationCancelled = true;
              }
              else
              {
                AccessFixedDimensionByItk_n(segmentationImage,
                    ITKPreprocessingOfSeedsForChangingSlice, 3,
                    (seeds,
                     sliceAxis,
                     d->m_SliceIndex,
                     sliceIndex,
                     false, // We propagate seeds at current position, so no optimisation
                     sliceIsEmpty,
                     *(copyOfCurrentSeeds.GetPointer()),
                     *(propagatedSeeds.GetPointer()),
                     outputRegion
                    )
                  );

                if (isThresholdingOn)
                {
                  QString message = tr("Thresholding slice %1 before copying marks to slice %2").arg(d->m_SliceIndex).arg(sliceIndex);
                  OpThresholdApply::ProcessorPointer processor = OpThresholdApply::ProcessorType::New();
                  doOp = new OpThresholdApply(OP_THRESHOLD_APPLY, true, outputRegion, processor, true);
                  undoOp = new OpThresholdApply(OP_THRESHOLD_APPLY, false, outputRegion, processor, true);
                  mitk::OperationEvent* operationEvent = new mitk::OperationEvent(d->m_Interface, doOp, undoOp, message.toStdString());
                  mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
                  this->ExecuteOperation(doOp);

                  drawTool->ClearWorkingData();
                  this->UpdateCurrentSliceContours();
                }

                // Do retain marks, which copies slice from beforeSliceIndex to afterSliceIndex
                QString message = tr("Retaining marks in slice %1 and copying to %2").arg(d->m_SliceIndex).arg(sliceIndex);
                OpRetainMarks::ProcessorPointer processor = OpRetainMarks::ProcessorType::New();
                doOp = new OpRetainMarks(OP_RETAIN_MARKS, true, d->m_SliceIndex, sliceIndex, sliceAxis, itkOrientation, outputRegion, processor);
                undoOp = new OpRetainMarks(OP_RETAIN_MARKS, false, d->m_SliceIndex, sliceIndex, sliceAxis, itkOrientation, outputRegion, processor);
                mitk::OperationEvent* operationEvent = new mitk::OperationEvent(d->m_Interface, doOp, undoOp, message.toStdString());
                mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
                this->ExecuteOperation(doOp);
              }
            }
            else // so, "Retain Marks" is Off.
            {
              AccessFixedDimensionByItk_n(segmentationImage,
                  ITKPreprocessingOfSeedsForChangingSlice, 3,
                  (seeds,
                   sliceAxis,
                   d->m_SliceIndex,
                   sliceIndex,
                   true, // optimise seed position on current slice.
                   sliceIsEmpty,
                   *(copyOfCurrentSeeds.GetPointer()),
                   *(propagatedSeeds.GetPointer()),
                   outputRegion
                  )
                );

              if (isThresholdingOn)
              {
                OpThresholdApply::ProcessorPointer processor = OpThresholdApply::ProcessorType::New();
                doOp = new OpThresholdApply(OP_THRESHOLD_APPLY, true, outputRegion, processor, true);
                undoOp = new OpThresholdApply(OP_THRESHOLD_APPLY, false, outputRegion, processor, true);
                mitk::OperationEvent* operationApplyEvent = new mitk::OperationEvent(d->m_Interface, doOp, undoOp, "Apply threshold");
                mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationApplyEvent );
                this->ExecuteOperation(doOp);

                drawTool->ClearWorkingData();
                this->UpdateCurrentSliceContours();
              }
              else // threshold box not checked
              {
                bool thisSliceHasUnenclosedSeeds = this->DoesSliceHaveUnenclosedSeeds(false, d->m_SliceIndex);

                if (thisSliceHasUnenclosedSeeds)
                {
                  OpWipe::ProcessorPointer processor = OpWipe::ProcessorType::New();
                  doOp = new OpWipe(OP_WIPE, true, sliceAxis, d->m_SliceIndex, outputRegion, propagatedSeeds, processor);
                  undoOp = new OpWipe(OP_WIPE, false, sliceAxis, d->m_SliceIndex, outputRegion, copyOfCurrentSeeds, processor);
                  mitk::OperationEvent* operationEvent = new mitk::OperationEvent(d->m_Interface, doOp, undoOp, "Wipe command");
                  mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
                  this->ExecuteOperation(doOp);
                }
                else // so, we don't have unenclosed seeds
                {
                  // There may be the case where the user has simply drawn a region, and put a seed in the middle.
                  // So, we do a region growing, without intensity limits. (we already know there are no unenclosed seeds).

                  this->UpdateRegionGrowing(false,
                                            d->m_SliceIndex,
                                            referenceImage->GetStatistics()->GetScalarValueMinNoRecompute(),
                                            referenceImage->GetStatistics()->GetScalarValueMaxNoRecompute(),
                                            false);

                  // Then we "apply" this region growing.
                  OpThresholdApply::ProcessorPointer processor = OpThresholdApply::ProcessorType::New();
                  doOp = new OpThresholdApply(OP_THRESHOLD_APPLY, true, outputRegion, processor, false);
                  undoOp = new OpThresholdApply(OP_THRESHOLD_APPLY, false, outputRegion, processor, false);
                  mitk::OperationEvent* operationApplyEvent = new mitk::OperationEvent(d->m_Interface, doOp, undoOp, "Apply threshold");
                  mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationApplyEvent );
                  this->ExecuteOperation(doOp);

                  drawTool->ClearWorkingData();

                } // end if/else unenclosed seeds
              } // end if/else thresholding on
            } // end if/else retain marks.

            if (!operationCancelled)
            {
              std::string orientationName = GetOrientationName(orientation);
              QString message = tr("Propagate seeds on %1 slice %2 (image axis: %3, slice: %4)")
                  .arg(QString::fromStdString(orientationName)).arg(selectedSliceIndex)
                  .arg(sliceAxis).arg(sliceIndex);
              doOp = new OpPropagateSeeds(OP_PROPAGATE_SEEDS, true, sliceAxis, sliceIndex, propagatedSeeds);
              undoOp = new OpPropagateSeeds(OP_PROPAGATE_SEEDS, false, sliceAxis, d->m_SliceIndex, copyOfCurrentSeeds);
              mitk::OperationEvent* operationPropEvent = new mitk::OperationEvent(d->m_Interface, doOp, undoOp, message.toStdString());
              mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationPropEvent );
              this->ExecuteOperation(doOp);

              message = tr("Change %1 slice from %2 to %3 (image axis: %4, from slice: %5 to slice: %6)")
                  .arg(QString::fromStdString(orientationName)).arg(d->m_SelectedSliceIndex).arg(selectedSliceIndex)
                  .arg(sliceAxis).arg(d->m_SliceIndex).arg(sliceIndex);
              doOp = new OpChangeSliceCommand(OP_CHANGE_SLICE, true, d->m_SelectedPosition, selectedPosition);
              undoOp = new OpChangeSliceCommand(OP_CHANGE_SLICE, false, d->m_SelectedPosition, selectedPosition);
              mitk::OperationEvent* operationEvent = new mitk::OperationEvent(d->m_Interface, doOp, undoOp, message.toStdString());
              mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
              this->ExecuteOperation(doOp);
            }
          }
          catch(const mitk::AccessByItkException& e)
          {
            MITK_ERROR << "Could not change slice: Caught mitk::AccessByItkException:" << e.what() << std::endl;
          }
          catch(const itk::ExceptionObject& e)
          {
            MITK_ERROR << "Could not change slice: Caught itk::ExceptionObject:" << e.what() << std::endl;
          }

          if (!operationCancelled)
          {
            if (MIDASPolyTool* polyTool = dynamic_cast<MIDASPolyTool*>(toolManager->GetActiveTool()))
            {
              //toolManager->ActivateTool(-1);
              /// This makes the poly tool save its result to the working data nodes and stay it open.
              polyTool->Deactivated();
              polyTool->Activated();
            }
          }

          d->m_IsUpdating = wasUpdating;

          this->UpdateCurrentSliceContours(false);
          this->UpdatePriorAndNext(false);
          this->UpdateRegionGrowing(false);
        }
        else // changing to any other slice (not the previous or next on the same orientation)
        {
          this->InitialiseSeedsForSlice(sliceAxis, sliceIndex);
          this->UpdateCurrentSliceContours(false);
          this->UpdatePriorAndNext(false);
          this->UpdateRegionGrowing(false);
          this->OnThresholdingCheckBoxToggled(isThresholdingOn);
        }

        this->RequestRenderWindowUpdate();

      } // if not being updated and not changing slice

      d->m_SliceIndex = sliceIndex;
      d->m_SelectedPosition = selectedPosition;

    } // if initialised and valid orientation (2D window selected)

    d->m_Orientation = orientation;
    d->m_SelectedSliceIndex = selectedSliceIndex;

  } // if orientation or selected slice has changed
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnNodeChanged(const mitk::DataNode* node)
{
  Q_D(GeneralSegmentorController);

  if (d->m_IsDeleting
      || d->m_IsUpdating
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

    if (workingData[MIDASTool::SEEDS] && workingData[MIDASTool::SEEDS] == node)
    {
      seedsChanged = true;
    }
    if (workingData[MIDASTool::DRAW_CONTOURS] && workingData[MIDASTool::DRAW_CONTOURS] == node)
    {
      drawContoursChanged = true;
    }

    if (!seedsChanged && !drawContoursChanged)
    {
      return;
    }

    mitk::DataNode::Pointer segmentationImageNode = workingData[MIDASTool::SEGMENTATION];
    if (segmentationImageNode.IsNotNull())
    {
      mitk::PointSet* seeds = this->GetSeeds();
      if (seeds && seeds->GetSize() > 0)
      {

        bool contourIsBeingEdited(false);
        if (segmentationImageNode.GetPointer() == node)
        {
          segmentationImageNode->GetBoolProperty(MIDASContourTool::EDITING_PROPERTY_NAME.c_str(), contourIsBeingEdited);
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
void GeneralSegmentorController::OnNodeRemoved(const mitk::DataNode* removedNode)
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::DataNode::Pointer segmentationNode = this->GetToolManager()->GetWorkingData(MIDASTool::SEGMENTATION);

  if (segmentationNode.GetPointer() == removedNode)
  {
    this->DiscardSegmentation();
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnContoursChanged()
{
  this->UpdateRegionGrowing();
}


//-----------------------------------------------------------------------------
mitk::PointSet* GeneralSegmentorController::GetSeeds()
{
  mitk::PointSet* result = nullptr;

  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::DataNode* seedsNode = toolManager->GetWorkingData(MIDASTool::SEEDS);
  if (seedsNode)
  {
    result = dynamic_cast<mitk::PointSet*>(seedsNode->GetData());
  }

  return result;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::InitialiseSeedsForSlice(int sliceAxis, int sliceIndex)
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::PointSet* seeds = this->GetSeeds();
  assert(seeds);

  mitk::Image::Pointer workingImage = this->GetWorkingImage(MIDASTool::SEGMENTATION);
  assert(workingImage);

  try
  {
    AccessFixedDimensionByItk_n(workingImage,
        ITKInitialiseSeedsForSlice, 3,
        (*seeds,
         sliceAxis,
         sliceIndex
        )
      );
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Caught exception during ITKInitialiseSeedsForSlice, so have not initialised seeds correctly, caused by:" << e.what();
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::RecalculateMinAndMaxOfImage()
{
  Q_D(GeneralSegmentorController);

  mitk::Image* referenceImage = this->GetReferenceImage();
  if (referenceImage)
  {
    double min = referenceImage->GetStatistics()->GetScalarValueMinNoRecompute();
    double max = referenceImage->GetStatistics()->GetScalarValueMaxNoRecompute();
    d->m_GUI->SetLowerAndUpperIntensityRanges(min, max);
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::RecalculateMinAndMaxOfSeedValues()
{
  Q_D(GeneralSegmentorController);

  mitk::Image* referenceImage = this->GetReferenceImage();
  mitk::PointSet* seeds = this->GetSeeds();

  if (referenceImage && seeds)
  {
    double min = 0;
    double max = 0;

    int sliceIndex = this->GetReferenceImageSliceIndex();
    int sliceAxis = this->GetReferenceImageSliceAxis();

    if (sliceIndex != -1 && sliceAxis != -1)
    {
      try
      {
        AccessFixedDimensionByItk_n(referenceImage, ITKRecalculateMinAndMaxOfSeedValues, 3, (seeds, sliceAxis, sliceIndex, min, max));
        d->m_GUI->SetSeedMinAndMaxValues(min, max);
      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Caught exception, so abandoning recalculating min and max of seeds values, due to:" << e.what();
      }
    }
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::UpdateCurrentSliceContours(bool updateRendering)
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  int sliceIndex = this->GetReferenceImageSliceIndex();
  int sliceAxis = this->GetReferenceImageSliceAxis();

  mitk::Image::Pointer workingImage = this->GetWorkingImage(MIDASTool::SEGMENTATION);
  assert(workingImage);

  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
  mitk::ContourModelSet::Pointer contourSet = dynamic_cast<mitk::ContourModelSet*>(workingData[MIDASTool::CONTOURS]->GetData());

  // TODO
  // This assertion fails sometimes if both the morphological and irregular (this) volume editor is
  // switched on and you are using the paintbrush tool of the morpho editor.
//  assert(contourSet);

  if (contourSet)
  {
    if (sliceIndex >= 0 && sliceAxis >= 0)
    {
      GenerateOutlineFromBinaryImage(workingImage, sliceAxis, sliceIndex, sliceIndex, contourSet);

      if (contourSet->GetSize() > 0)
      {
        workingData[MIDASTool::CONTOURS]->Modified();

        if (updateRendering)
        {
          this->RequestRenderWindowUpdate();
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnSeePriorCheckBoxToggled(bool checked)
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
  workingData[MIDASTool::PRIOR_CONTOURS]->SetVisibility(checked);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnSeeNextCheckBoxToggled(bool checked)
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
  workingData[MIDASTool::NEXT_CONTOURS]->SetVisibility(checked);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnThresholdingCheckBoxToggled(bool checked)
{
  Q_D(GeneralSegmentorController);

  if (!this->HasInitialisedWorkingData())
  {
    // So, if there is NO working data, we leave the widgets disabled regardless.
    d->m_GUI->SetThresholdingWidgetsEnabled(false);
    return;
  }

  this->RecalculateMinAndMaxOfImage();
  this->RecalculateMinAndMaxOfSeedValues();

  d->m_GUI->SetThresholdingWidgetsEnabled(checked);

  if (checked)
  {
    this->UpdateRegionGrowing();
  }

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
  workingData[MIDASTool::REGION_GROWING]->SetVisibility(checked);

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnThresholdValueChanged()
{
  this->UpdateRegionGrowing();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::UpdateRegionGrowing(bool updateRendering)
{
  Q_D(GeneralSegmentorController);

  bool isThresholdingOn = d->m_GUI->IsThresholdingCheckBoxChecked();

  if (isThresholdingOn)
  {
    int sliceIndex = this->GetReferenceImageSliceIndex();
    double lowerThreshold = d->m_GUI->GetLowerThreshold();
    double upperThreshold = d->m_GUI->GetUpperThreshold();
    bool skipUpdate = !isThresholdingOn;

    this->UpdateRegionGrowing(isThresholdingOn, sliceIndex, lowerThreshold, upperThreshold, skipUpdate);

    if (updateRendering)
    {
      this->RequestRenderWindowUpdate();
    }
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::UpdateRegionGrowing(
    bool isVisible,
    int sliceIndex,
    double lowerThreshold,
    double upperThreshold,
    bool skipUpdate
    )
{
  Q_D(GeneralSegmentorController);

  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::Image* referenceImage = this->GetReferenceImage();
  if (referenceImage)
  {
    mitk::DataNode::Pointer segmentationNode = this->GetWorkingData()[MIDASTool::SEGMENTATION];
    mitk::Image::Pointer segmentationImage = this->GetWorkingImage(MIDASTool::SEGMENTATION);

    if (segmentationImage.IsNotNull() && segmentationNode.IsNotNull())
    {

      mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
      workingData[MIDASTool::REGION_GROWING]->SetVisibility(isVisible);

      bool wasUpdating = d->m_IsUpdating;
      d->m_IsUpdating = true;

      mitk::DataNode::Pointer regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(MIDASTool::REGION_GROWING_NAME.c_str(), segmentationNode, true);
      assert(regionGrowingNode);

      mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
      assert(regionGrowingImage);

      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      mitk::ToolManager *toolManager = this->GetToolManager();
      assert(toolManager);

      MIDASPolyTool *polyTool = this->GetToolByType<MIDASPolyTool>();
      assert(polyTool);

      mitk::ContourModelSet::Pointer polyToolContours = mitk::ContourModelSet::New();

      mitk::ContourModel* polyToolContour = polyTool->GetContour();
      if (polyToolContour && polyToolContour->GetNumberOfVertices() >= 2)
      {
        polyToolContours->AddContourModel(polyToolContour);
      }

      mitk::ContourModelSet* segmentationContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[MIDASTool::CONTOURS]->GetData());
      mitk::ContourModelSet* drawToolContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[MIDASTool::DRAW_CONTOURS]->GetData());

      int sliceAxis = this->GetReferenceImageSliceAxis();

      if (sliceAxis != -1 && sliceIndex != -1)
      {
        try
        {
          AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
              ITKUpdateRegionGrowing, 3,
              (skipUpdate,
               segmentationImage,
               seeds,
               segmentationContours,
               drawToolContours,
               polyToolContours,
               sliceAxis,
               sliceIndex,
               lowerThreshold,
               upperThreshold,
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
        MITK_ERROR << "Could not do region growing: Error sliceAxis=" << sliceAxis << ", sliceIndex=" << sliceIndex << std::endl;
      }

      d->m_IsUpdating = wasUpdating;
    }
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::UpdatePriorAndNext(bool updateRendering)
{
  Q_D(GeneralSegmentorController);

  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  int sliceIndex = this->GetReferenceImageSliceIndex();
  int sliceAxis = this->GetReferenceImageSliceAxis();

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
  mitk::Image::Pointer segmentationImage = this->GetWorkingImage(MIDASTool::SEGMENTATION);

  if (d->m_GUI->IsSeePriorCheckBoxChecked())
  {
    mitk::ContourModelSet::Pointer contourSet = dynamic_cast<mitk::ContourModelSet*>(workingData[MIDASTool::PRIOR_CONTOURS]->GetData());
    GenerateOutlineFromBinaryImage(segmentationImage, sliceAxis, sliceIndex-1, sliceIndex, contourSet);

    if (contourSet->GetSize() > 0)
    {
      workingData[MIDASTool::PRIOR_CONTOURS]->Modified();

      if (updateRendering)
      {
        this->RequestRenderWindowUpdate();
      }
    }
  }

  if (d->m_GUI->IsSeeNextCheckBoxChecked())
  {
    mitk::ContourModelSet::Pointer contourSet = dynamic_cast<mitk::ContourModelSet*>(workingData[MIDASTool::NEXT_CONTOURS]->GetData());
    GenerateOutlineFromBinaryImage(segmentationImage, sliceAxis, sliceIndex+1, sliceIndex, contourSet);

    if (contourSet->GetSize() > 0)
    {
      workingData[MIDASTool::NEXT_CONTOURS]->Modified();

      if (updateRendering)
      {
        this->RequestRenderWindowUpdate();
      }
    }
  }
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::DoesSliceHaveUnenclosedSeeds(bool thresholdOn, int sliceIndex)
{
  mitk::PointSet* seeds = this->GetSeeds();
  assert(seeds);

  return this->DoesSliceHaveUnenclosedSeeds(thresholdOn, sliceIndex, seeds);
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::DoesSliceHaveUnenclosedSeeds(bool thresholdOn, int sliceIndex, const mitk::PointSet* seeds)
{
  Q_D(GeneralSegmentorController);

  bool sliceDoesHaveUnenclosedSeeds = false;

  if (!this->HasInitialisedWorkingData())
  {
    return sliceDoesHaveUnenclosedSeeds;
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImage();
  mitk::Image::Pointer segmentationImage = this->GetWorkingImage(MIDASTool::SEGMENTATION);

  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  MIDASPolyTool* polyTool = this->GetToolByType<MIDASPolyTool>();
  assert(polyTool);

  mitk::ContourModelSet::Pointer polyToolContours = mitk::ContourModelSet::New();
  mitk::ContourModel* polyToolContour = polyTool->GetContour();
  if (polyToolContour && polyToolContour->GetNumberOfVertices() >= 2)
  {
    polyToolContours->AddContourModel(polyToolContour);
  }

  mitk::ContourModelSet* segmentationContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[MIDASTool::CONTOURS]->GetData());
  mitk::ContourModelSet* drawToolContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[MIDASTool::DRAW_CONTOURS]->GetData());

  double lowerThreshold = d->m_GUI->GetLowerThreshold();
  double upperThreshold = d->m_GUI->GetUpperThreshold();

  int sliceAxis = this->GetReferenceImageSliceAxis();

  if (sliceAxis != -1 && sliceIndex != -1)
  {
    try
    {
      AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
        ITKSliceDoesHaveUnenclosedSeeds, 3,
          (seeds,
           *segmentationContours,
           *polyToolContours,
           *drawToolContours,
           segmentationImage,
           lowerThreshold,
           upperThreshold,
           thresholdOn,
           sliceAxis,
           sliceIndex,
           sliceDoesHaveUnenclosedSeeds
          )
      );
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception during ITKSliceDoesHaveUnenclosedSeeds, so will return false, caused by:" << e.what();
    }
  }

  return sliceDoesHaveUnenclosedSeeds;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::FilterSeedsToCurrentSlice(
    const mitk::PointSet* inputPoints,
    int sliceAxis,
    int sliceIndex,
    mitk::PointSet& outputPoints
    )
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::Image::Pointer referenceImage = this->GetReferenceImage();
  if (referenceImage.IsNotNull())
  {
    try
    {
      AccessFixedDimensionByItk_n(referenceImage,
          ITKFilterSeedsToCurrentSlice, 3,
          (inputPoints,
           sliceAxis,
           sliceIndex,
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
void GeneralSegmentorController::FilterSeedsToEnclosedSeedsOnCurrentSlice(
    mitk::PointSet& inputPoints,
    bool& thresholdOn,
    int& sliceIndex,
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

    bool unenclosed = this->DoesSliceHaveUnenclosedSeeds(thresholdOn, sliceIndex, singleSeedPointSet);

    if (!unenclosed)
    {
      outputPoints.InsertPoint(pointID, point);
    }
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnAnyButtonClicked()
{
  /// Set the focus back to the main window. This is needed so that the keyboard shortcuts
  /// (like 'a' and 'z' for changing slice) keep on working.
  if (QmitkRenderWindow* mainWindow = this->GetView()->GetSelectedRenderWindow())
  {
    mainWindow->setFocus();
  }
}


/**************************************************************
 * Start of: Functions for OK/Reset/Cancel/Close.
 * i.e. finishing a segmentation, and destroying stuff.
 *************************************************************/

//-----------------------------------------------------------------------------
void GeneralSegmentorController::DestroyPipeline()
{
  Q_D(GeneralSegmentorController);

  mitk::Image* referenceImage = this->GetReferenceImage();
  if (referenceImage)
  {
    bool wasDeleting = d->m_IsDeleting;
    d->m_IsDeleting = true;
    try
    {
      AccessFixedDimensionByItk(referenceImage, ITKDestroyPipeline, 3);
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so abandoning destroying the ITK pipeline, caused by:" << e.what();
    }
    d->m_IsDeleting = wasDeleting;
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::RemoveWorkingData()
{
  Q_D(GeneralSegmentorController);

  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  bool wasDeleting = d->m_IsDeleting;
  d->m_IsDeleting = true;

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

  d->m_IsDeleting = wasDeleting;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::RestoreInitialSegmentation()
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::DataNode::Pointer segmentationNode = this->GetToolManager()->GetWorkingData(MIDASTool::SEGMENTATION);
  assert(segmentationNode);

  mitk::DataNode::Pointer seedsNode = this->GetToolManager()->GetWorkingData(MIDASTool::SEEDS);
  assert(seedsNode);

  try
  {
    /// Originally, this function cleared the segmentation and the pointset, but
    /// now we rather restore the initial state of the segmentation as it was
    /// when we pressed the Create/restart segmentation button.

//    mitk::Image::Pointer segmentationImage = dynamic_cast<mitk::Image*>(segmentationNode->GetData());
//    assert(segmentationImage);
//    AccessFixedDimensionByItk(segmentationImage.GetPointer(), ITKClearImage, 3);
//    segmentationImage->Modified();
//    segmentationNode->Modified();

//    mitk::PointSet::Pointer seeds = this->GetSeeds();
//    seeds->Clear();

    mitk::DataNode::Pointer initialSegmentationNode = this->GetToolManager()->GetWorkingData(MIDASTool::INITIAL_SEGMENTATION);
    mitk::DataNode::Pointer initialSeedsNode = this->GetToolManager()->GetWorkingData(MIDASTool::INITIAL_SEEDS);

    segmentationNode->SetData(dynamic_cast<mitk::Image*>(initialSegmentationNode->GetData())->Clone());
    seedsNode->SetData(dynamic_cast<mitk::PointSet*>(initialSeedsNode->GetData())->Clone());

    this->UpdateCurrentSliceContours(false);
    this->UpdateRegionGrowing(false);
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Caught exception during ITKClearImage, caused by:" << e.what();
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnOKButtonClicked()
{
  Q_D(GeneralSegmentorController);

  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  // Set the colour to that which the user selected in the first place.
  mitk::DataNode::Pointer workingData = this->GetToolManager()->GetWorkingData(MIDASTool::SEGMENTATION);
  workingData->SetProperty("color", workingData->GetProperty("midas.tmp.selectedcolor"));
  workingData->SetProperty("binaryimage.selectedcolor", workingData->GetProperty("midas.tmp.selectedcolor"));

  /// Apply the thresholds if we are thresholding, and chunk out the contour segments that
  /// do not close any region with seed.
  this->OnCleanButtonClicked();

  this->DestroyPipeline();
  this->RemoveWorkingData();
  d->m_GUI->EnableSegmentationWidgets(false);
  this->GetView()->SetCurrentSelection(workingData);

  this->RequestRenderWindowUpdate();
  mitk::UndoController::GetCurrentUndoModel()->Clear();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnResetButtonClicked()
{
  Q_D(GeneralSegmentorController);

  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  int returnValue = QMessageBox::warning(d->m_GUI->GetParent(), tr("NiftyMIDAS"),
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
void GeneralSegmentorController::OnCancelButtonClicked()
{
  this->DiscardSegmentation();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnViewGetsClosed()
{
  /// TODO this is not invoked at all.
  /// This function was called "ClosePart" before it was moved here from niftkGeneralSegmentorView.
  /// It was not invoked there, either. I leave this here to remind me that the segmentation should
  /// be discarded when the view is closed.
  this->DiscardSegmentation();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::DiscardSegmentation()
{
  Q_D(GeneralSegmentorController);

  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::DataNode::Pointer segmentationNode = this->GetToolManager()->GetWorkingData(MIDASTool::SEGMENTATION);
  assert(segmentationNode);

  this->DestroyPipeline();
  if (d->m_IsRestarting)
  {
    this->RestoreInitialSegmentation();
    this->RemoveWorkingData();
  }
  else
  {
    this->RemoveWorkingData();
    this->GetDataStorage()->Remove(segmentationNode);
  }
  d->m_GUI->EnableSegmentationWidgets(false);
  this->SetReferenceImageSelected();
  this->RequestRenderWindowUpdate();
  mitk::UndoController::GetCurrentUndoModel()->Clear();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnRestartButtonClicked()
{
  Q_D(GeneralSegmentorController);

  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  int returnValue = QMessageBox::warning(d->m_GUI->GetParent(), tr("NiftyMIDAS"),
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
void GeneralSegmentorController::ClearWorkingData()
{
  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  mitk::DataNode::Pointer workingData = this->GetToolManager()->GetWorkingData(MIDASTool::SEGMENTATION);
  assert(workingData);

  mitk::Image::Pointer segmentationImage = dynamic_cast<mitk::Image*>(workingData->GetData());
  assert(segmentationImage);

  try
  {
    AccessFixedDimensionByItk(segmentationImage.GetPointer(), ITKClearImage, 3);
    segmentationImage->Modified();
    workingData->Modified();

    mitk::PointSet::Pointer seeds = this->GetSeeds();
    seeds->Clear();

    this->UpdateCurrentSliceContours(false);
    this->UpdateRegionGrowing(false);
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Caught exception during ITKClearImage, caused by:" << e.what();
  }
}


/**************************************************************
 * End of: Functions for OK/Reset/Cancel/Close.
 *************************************************************/

/**************************************************************
 * Start of: Functions for simply tool toggling
 *************************************************************/

//-----------------------------------------------------------------------------
void GeneralSegmentorController::ToggleTool(int toolId)
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
bool GeneralSegmentorController::SelectSeedTool()
{
  Q_D(GeneralSegmentorController);

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

  if (d->m_GUI->IsToolSelectorEnabled())
  {
    mitk::ToolManager* toolManager = this->GetToolManager();
    int activeToolId = toolManager->GetActiveToolID();
    int seedToolId = toolManager->GetToolIdByToolType<MIDASSeedTool>();

    if (seedToolId != activeToolId)
    {
      toolManager->ActivateTool(seedToolId);
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::SelectDrawTool()
{
  Q_D(GeneralSegmentorController);

  /// Note: see comment in SelectSeedTool().
  if (d->m_GUI->IsToolSelectorEnabled())
  {
    mitk::ToolManager* toolManager = this->GetToolManager();
    int activeToolId = toolManager->GetActiveToolID();
    int drawToolId = toolManager->GetToolIdByToolType<MIDASDrawTool>();

    if (drawToolId != activeToolId)
    {
      toolManager->ActivateTool(drawToolId);
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::SelectPolyTool()
{
  Q_D(GeneralSegmentorController);

  /// Note: see comment in SelectSeedTool().
  if (d->m_GUI->IsToolSelectorEnabled())
  {
    mitk::ToolManager* toolManager = this->GetToolManager();
    int activeToolId = toolManager->GetActiveToolID();
    int polyToolId = toolManager->GetToolIdByToolType<MIDASPolyTool>();

    if (polyToolId != activeToolId)
    {
      toolManager->ActivateTool(polyToolId);
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::UnselectTools()
{
  Q_D(GeneralSegmentorController);

  if (d->m_GUI->IsToolSelectorEnabled())
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
bool GeneralSegmentorController::SelectViewMode()
{
  Q_D(GeneralSegmentorController);

  /// Note: see comment in SelectSeedTool().
  if (d->m_GUI->IsToolSelectorEnabled())
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
    bool segmentationNodeIsVisible = workingData[MIDASTool::SEGMENTATION]->IsVisible(0);
    workingData[MIDASTool::SEGMENTATION]->SetVisibility(!segmentationNodeIsVisible);
    this->RequestRenderWindowUpdate();

    return true;
  }

  return false;
}


/**************************************************************
 * End of: Functions for simply tool toggling
 *************************************************************/

//-----------------------------------------------------------------------------
bool GeneralSegmentorController::CleanSlice()
{
  Q_D(GeneralSegmentorController);

  /// Note: see comment in SelectSeedTool().
  if (d->m_GUI->IsToolSelectorEnabled())
  {
    this->OnCleanButtonClicked();
    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnPropagate3DButtonClicked()
{
  this->DoPropagate(false, true);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnPropagateUpButtonClicked()
{
  this->DoPropagate(true, false);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnPropagateDownButtonClicked()
{
  this->DoPropagate(false, false);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::DoPropagate(bool isUp, bool is3D)
{
  Q_D(GeneralSegmentorController);

  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  ImageOrientation imageOrientation = this->GetOrientation();
  itk::Orientation orientation = GetItkOrientation(imageOrientation);

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
      if (imageOrientation == IMAGE_ORIENTATION_AXIAL)
      {
        orientationText = "superior to";
      }
      else if (imageOrientation == IMAGE_ORIENTATION_SAGITTAL)
      {
        orientationText = "right of";
      }
      else if (imageOrientation == IMAGE_ORIENTATION_CORONAL)
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
      if (imageOrientation == IMAGE_ORIENTATION_AXIAL)
      {
        orientationText = "inferior to";
      }
      else if (imageOrientation == IMAGE_ORIENTATION_SAGITTAL)
      {
        orientationText = "left of";
      }
      else if (imageOrientation == IMAGE_ORIENTATION_CORONAL)
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

  int returnValue = QMessageBox::warning(d->m_GUI->GetParent(), tr("NiftyMIDAS"),
                                                   tr("%1.\n"
                                                      "Are you sure?").arg(message),
                                                   QMessageBox::Yes | QMessageBox::No);
  if (returnValue == QMessageBox::No)
  {
    return;
  }

  mitk::Image* referenceImage = this->GetReferenceImage();
  if (referenceImage)
  {

    mitk::DataNode::Pointer segmentationNode = this->GetWorkingData()[MIDASTool::SEGMENTATION];
    mitk::Image::Pointer segmentationImage = this->GetWorkingImage(MIDASTool::SEGMENTATION);

    if (segmentationImage.IsNotNull() && segmentationNode.IsNotNull())
    {

      mitk::DataNode::Pointer regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(MIDASTool::REGION_GROWING_NAME.c_str(), segmentationNode, true);
      assert(regionGrowingNode);

      mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
      assert(regionGrowingImage);

      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      mitk::ToolManager* toolManager = this->GetToolManager();
      assert(toolManager);

      MIDASDrawTool* drawTool = this->GetToolByType<MIDASDrawTool>();
      assert(drawTool);

      double lowerThreshold = d->m_GUI->GetLowerThreshold();
      double upperThreshold = d->m_GUI->GetUpperThreshold();
      int sliceAxis = this->GetReferenceImageSliceAxis();
      int sliceIndex = this->GetReferenceImageSliceIndex();
      int sliceUpDirection = this->GetReferenceImageSliceUpDirection();
      if (!is3D && !isUp)
      {
        sliceUpDirection *= -1;
      }
      else if (is3D)
      {
        sliceUpDirection = 0;
      }

      mitk::PointSet::Pointer copyOfInputSeeds = mitk::PointSet::New();
      mitk::PointSet::Pointer outputSeeds = mitk::PointSet::New();
      std::vector<int> outputRegion;

      if (sliceAxis != -1 && sliceIndex != -1 && orientation != itk::ORIENTATION_UNKNOWN)
      {
        bool wasUpdating = d->m_IsUpdating;
        d->m_IsUpdating = true;

        try
        {
          AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
              ITKPropagateToRegionGrowingImage, 3,
              (seeds,
               sliceAxis,
               sliceIndex,
               sliceUpDirection,
               lowerThreshold,
               upperThreshold,
               *(copyOfInputSeeds.GetPointer()),
               *(outputSeeds.GetPointer()),
               outputRegion,
               regionGrowingImage  // This is the image we are writing to.
              )
            );

          if (toolManager->GetActiveToolID() == toolManager->GetToolIdByToolType<MIDASPolyTool>())
          {
            toolManager->ActivateTool(-1);
          }

          mitk::UndoStackItem::IncCurrObjectEventId();
          mitk::UndoStackItem::IncCurrGroupEventId();
          mitk::UndoStackItem::ExecuteIncrement();

          QString message = tr("Propagate: copy region growing");
          OpPropagate::ProcessorPointer processor = OpPropagate::ProcessorType::New();
          OpPropagate *doPropOp = new OpPropagate(OP_PROPAGATE, true, outputRegion, processor);
          OpPropagate *undoPropOp = new OpPropagate(OP_PROPAGATE, false, outputRegion, processor);
          mitk::OperationEvent* operationEvent = new mitk::OperationEvent(d->m_Interface, doPropOp, undoPropOp, message.toStdString());
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
          this->ExecuteOperation(doPropOp);

          message = tr("Propagate: copy seeds");
          OpPropagateSeeds *doPropSeedsOp = new OpPropagateSeeds(OP_PROPAGATE_SEEDS, true, sliceAxis, sliceIndex, outputSeeds);
          OpPropagateSeeds *undoPropSeedsOp = new OpPropagateSeeds(OP_PROPAGATE_SEEDS, false, sliceAxis, sliceIndex, copyOfInputSeeds);
          mitk::OperationEvent* operationPropEvent = new mitk::OperationEvent(d->m_Interface, doPropSeedsOp, undoPropSeedsOp, message.toStdString());
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
        catch(const itk::ExceptionObject& e)
        {
          MITK_ERROR << "Could not propagate: Caught itk::ExceptionObject:" << e.what() << std::endl;
        }

        d->m_IsUpdating = wasUpdating;
      }
      else
      {
        MITK_ERROR << "Could not propagate: Error sliceAxis=" << sliceAxis << ", sliceIndex=" << sliceIndex << ", orientation=" << orientation << ", direction=" << sliceUpDirection << std::endl;
      }
    }
  }

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnWipeButtonClicked()
{
  this->DoWipe(0);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnWipePlusButtonClicked()
{
  Q_D(GeneralSegmentorController);

  ImageOrientation imageOrientation = this->GetOrientation();

  QString orientationText;
  QString messageWithOrientation = "All slices %1 the present will be cleared \nAre you sure?";

  if (imageOrientation == IMAGE_ORIENTATION_AXIAL)
  {
    orientationText = "superior to";
  }
  else if (imageOrientation == IMAGE_ORIENTATION_SAGITTAL)
  {
    orientationText = "right of";
  }
  else if (imageOrientation == IMAGE_ORIENTATION_CORONAL)
  {
    orientationText = "anterior to";
  }
  else
  {
    orientationText = "up from";
  }

  int returnValue = QMessageBox::warning(d->m_GUI->GetParent(), tr("NiftyMIDAS"),
                                                            tr(messageWithOrientation.toStdString().c_str()).arg(orientationText),
                                                            QMessageBox::Yes | QMessageBox::No);
  if (returnValue == QMessageBox::No)
  {
    return;
  }

  this->DoWipe(1);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorController::OnWipeMinusButtonClicked()
{
  Q_D(GeneralSegmentorController);

  ImageOrientation imageOrientation = this->GetOrientation();

  QString orientationText;
  QString messageWithOrientation = "All slices %1 the present will be cleared \nAre you sure?";

  if (imageOrientation == IMAGE_ORIENTATION_AXIAL)
  {
    orientationText = "inferior to";
  }
  else if (imageOrientation == IMAGE_ORIENTATION_SAGITTAL)
  {
    orientationText = "left of";
  }
  else if (imageOrientation == IMAGE_ORIENTATION_CORONAL)
  {
    orientationText = "posterior to";
  }
  else
  {
    orientationText = "down from";
  }

  int returnValue = QMessageBox::warning(d->m_GUI->GetParent(), tr("NiftyMIDAS"),
                                                            tr(messageWithOrientation.toStdString().c_str()).arg(orientationText),
                                                            QMessageBox::Yes | QMessageBox::No);
  if (returnValue == QMessageBox::No)
  {
    return;
  }

  this->DoWipe(-1);
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::DoWipe(int direction)
{
  Q_D(GeneralSegmentorController);

  bool wipeWasPerformed = false;

  if (!this->HasInitialisedWorkingData())
  {
    return wipeWasPerformed;
  }

  mitk::Image* referenceImage = this->GetReferenceImage();
  if (referenceImage)
  {

    mitk::DataNode::Pointer segmentationNode = this->GetWorkingData()[MIDASTool::SEGMENTATION];
    mitk::Image::Pointer segmentationImage = this->GetWorkingImage(MIDASTool::SEGMENTATION);

    if (segmentationImage.IsNotNull() && segmentationNode.IsNotNull())
    {
      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      int sliceAxis = this->GetReferenceImageSliceAxis();
      int sliceIndex = this->GetReferenceImageSliceIndex();
      int sliceUpDirection = this->GetReferenceImageSliceUpDirection();

      if (direction != 0) // zero means, current slice.
      {
        direction *= sliceUpDirection;
      }

      mitk::PointSet::Pointer copyOfInputSeeds = mitk::PointSet::New();
      mitk::PointSet::Pointer outputSeeds = mitk::PointSet::New();
      std::vector<int> outputRegion;

      if (sliceAxis != -1 && sliceIndex != -1)
      {
        bool wasUpdating = d->m_IsUpdating;
        d->m_IsUpdating = true;

        try
        {

          mitk::ToolManager* toolManager = this->GetToolManager();
          assert(toolManager);

          MIDASDrawTool* drawTool = this->GetToolByType<MIDASDrawTool>();
          assert(drawTool);

          if (toolManager->GetActiveToolID() == toolManager->GetToolIdByToolType<MIDASPolyTool>())
          {
            toolManager->ActivateTool(-1);
          }


          if (direction == 0)
          {
            mitk::CopyPointSets(*seeds, *copyOfInputSeeds);
            mitk::CopyPointSets(*seeds, *outputSeeds);

            AccessFixedDimensionByItk_n(segmentationImage,
                ITKCalculateSliceRegionAsVector, 3,
                (sliceAxis,
                 sliceIndex,
                 outputRegion
                )
              );

          }
          else
          {
            AccessFixedDimensionByItk_n(segmentationImage, // The binary image = current segmentation
                ITKPreprocessingForWipe, 3,
                (seeds,
                 sliceAxis,
                 sliceIndex,
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

          OpWipe::ProcessorPointer processor = OpWipe::ProcessorType::New();
          OpWipe *doOp = new OpWipe(OP_WIPE, true, sliceAxis, sliceIndex, outputRegion, outputSeeds, processor);
          OpWipe *undoOp = new OpWipe(OP_WIPE, false, sliceAxis, sliceIndex, outputRegion, copyOfInputSeeds, processor);
          mitk::OperationEvent* operationEvent = new mitk::OperationEvent(d->m_Interface, doOp, undoOp, "Wipe command");
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
        catch(const itk::ExceptionObject& e)
        {
          MITK_ERROR << "Could not do wipe command: Caught itk::ExceptionObject:" << e.what() << std::endl;
        }

        d->m_IsUpdating = wasUpdating;
      }
      else
      {
        MITK_ERROR << "Could not wipe: Error, sliceAxis=" << sliceAxis << ", sliceIndex=" << sliceIndex << std::endl;
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
void GeneralSegmentorController::OnThresholdApplyButtonClicked()
{
  this->DoThresholdApply(true, false, false);
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorController::DoThresholdApply(
    bool optimiseSeeds,
    bool newSliceEmpty,
    bool newCheckboxStatus)
{
  Q_D(GeneralSegmentorController);

  if (!this->HasInitialisedWorkingData())
  {
    return false;
  }

  bool updateWasApplied = false;

  mitk::Image* referenceImage = this->GetReferenceImage();
  if (referenceImage)
  {
    ImageOrientation orientation = this->GetOrientation();
    int selectedSliceIndex = this->GetSliceIndex();

    int sliceIndex = this->GetReferenceImageSliceIndex();

    mitk::DataNode::Pointer segmentationNode = this->GetWorkingData()[MIDASTool::SEGMENTATION];
    mitk::Image::Pointer segmentationImage = this->GetWorkingImage(MIDASTool::SEGMENTATION);

    if (segmentationImage.IsNotNull() && segmentationNode.IsNotNull())
    {
      mitk::DataNode::Pointer regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(MIDASTool::REGION_GROWING_NAME.c_str(), segmentationNode, true);
      assert(regionGrowingNode);

      mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
      assert(regionGrowingImage);

      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      mitk::ToolManager* toolManager = this->GetToolManager();
      assert(toolManager);

      MIDASDrawTool* drawTool = this->GetToolByType<MIDASDrawTool>();
      assert(drawTool);

      int sliceAxis = this->GetReferenceImageSliceAxis();

      mitk::PointSet::Pointer copyOfInputSeeds = mitk::PointSet::New();
      mitk::PointSet::Pointer outputSeeds = mitk::PointSet::New();
      std::vector<int> outputRegion;

      if (sliceAxis != -1 && sliceIndex != -1)
      {
        bool wasUpdating = d->m_IsUpdating;
        d->m_IsUpdating = true;

        try
        {
          AccessFixedDimensionByItk_n(regionGrowingImage,
              ITKPreprocessingOfSeedsForChangingSlice, 3,
              (seeds,
               sliceAxis,
               sliceIndex,
               sliceIndex,
               optimiseSeeds,
               newSliceEmpty,
               *(copyOfInputSeeds.GetPointer()),
               *(outputSeeds.GetPointer()),
               outputRegion
              )
            );

          bool isThresholdingOn = d->m_GUI->IsThresholdingCheckBoxChecked();

          if (toolManager->GetActiveToolID() == toolManager->GetToolIdByToolType<MIDASPolyTool>())
          {
            toolManager->ActivateTool(-1);
          }

          mitk::UndoStackItem::IncCurrObjectEventId();
          mitk::UndoStackItem::IncCurrGroupEventId();
          mitk::UndoStackItem::ExecuteIncrement();

          std::string orientationName = GetOrientationName(orientation);
          QString message = tr("Apply threshold on %1 slice %2 (image axis: %3, slice: %4)")
              .arg(QString::fromStdString(orientationName)).arg(selectedSliceIndex)
              .arg(sliceAxis).arg(sliceIndex);
          OpThresholdApply::ProcessorPointer processor = OpThresholdApply::ProcessorType::New();
          OpThresholdApply *doThresholdOp = new OpThresholdApply(OP_THRESHOLD_APPLY, true, outputRegion, processor, newCheckboxStatus);
          OpThresholdApply *undoThresholdOp = new OpThresholdApply(OP_THRESHOLD_APPLY, false, outputRegion, processor, isThresholdingOn);
          mitk::OperationEvent* operationEvent = new mitk::OperationEvent(d->m_Interface, doThresholdOp, undoThresholdOp, message.toStdString());
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
          this->ExecuteOperation(doThresholdOp);

          message = tr("Propagate seeds on %1 slice %2 (image axis: %3, slice: %4)")
              .arg(QString::fromStdString(orientationName)).arg(selectedSliceIndex)
              .arg(sliceAxis).arg(sliceIndex);
          OpPropagateSeeds *doPropOp = new OpPropagateSeeds(OP_PROPAGATE_SEEDS, true, sliceAxis, sliceIndex, outputSeeds);
          OpPropagateSeeds *undoPropOp = new OpPropagateSeeds(OP_PROPAGATE_SEEDS, false, sliceAxis, sliceIndex, copyOfInputSeeds);
          mitk::OperationEvent* operationPropEvent = new mitk::OperationEvent(d->m_Interface, doPropOp, undoPropOp, message.toStdString());
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
        catch(const itk::ExceptionObject& e)
        {
          MITK_ERROR << "Could not do threshold apply command: Caught itk::ExceptionObject:" << e.what() << std::endl;
        }

        d->m_IsUpdating = wasUpdating;

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
void GeneralSegmentorController::OnCleanButtonClicked()
{
  Q_D(GeneralSegmentorController);

  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  bool isThresholdingOn = d->m_GUI->IsThresholdingCheckBoxChecked();
  int sliceIndex = this->GetReferenceImageSliceIndex();

  if (!isThresholdingOn)
  {
    bool hasUnenclosedSeeds = this->DoesSliceHaveUnenclosedSeeds(isThresholdingOn, sliceIndex);
    if (hasUnenclosedSeeds)
    {
      int returnValue = QMessageBox::warning(d->m_GUI->GetParent(), tr("NiftyMIDAS"),
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

  mitk::Image* referenceImage = this->GetReferenceImage();
  if (referenceImage)
  {
    mitk::DataNode::Pointer segmentationNode = this->GetWorkingData()[MIDASTool::SEGMENTATION];
    mitk::Image::Pointer segmentationImage = this->GetWorkingImage(MIDASTool::SEGMENTATION);

    if (segmentationImage.IsNotNull() && segmentationNode.IsNotNull())
    {
      mitk::PointSet* seeds = this->GetSeeds();
      assert(seeds);

      mitk::ToolManager* toolManager = this->GetToolManager();
      assert(toolManager);

      MIDASPolyTool* polyTool = this->GetToolByType<MIDASPolyTool>();
      assert(polyTool);

      MIDASDrawTool* drawTool = this->GetToolByType<MIDASDrawTool>();
      assert(drawTool);

      mitk::ContourModelSet::Pointer polyToolContours = mitk::ContourModelSet::New();

      mitk::ContourModel* polyToolContour = polyTool->GetContour();
      if (polyToolContour && polyToolContour->GetNumberOfVertices() >= 2)
      {
        polyToolContours->AddContourModel(polyToolContour);
      }

      mitk::ContourModelSet* segmentationContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[MIDASTool::CONTOURS]->GetData());
      assert(segmentationContours);

      mitk::ContourModelSet* drawToolContours = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[MIDASTool::DRAW_CONTOURS]->GetData());
      assert(drawToolContours);

      mitk::DataNode::Pointer regionGrowingNode = this->GetDataStorage()->GetNamedDerivedNode(MIDASTool::REGION_GROWING_NAME.c_str(), segmentationNode, true);
      assert(regionGrowingNode);

      mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());
      assert(regionGrowingImage);

      double lowerThreshold = d->m_GUI->GetLowerThreshold();
      double upperThreshold = d->m_GUI->GetUpperThreshold();
      int sliceAxis = this->GetReferenceImageSliceAxis();

      mitk::ContourModelSet::Pointer copyOfInputContourSet = mitk::ContourModelSet::New();
      mitk::ContourModelSet::Pointer outputContourSet = mitk::ContourModelSet::New();

      if (sliceAxis != -1 && sliceIndex != -1)
      {
        bool wasUpdating = d->m_IsUpdating;
        d->m_IsUpdating = true;

        try
        {
          // Calculate the region of interest for this slice.
          std::vector<int> outputRegion;
          AccessFixedDimensionByItk_n(segmentationImage,
              ITKCalculateSliceRegionAsVector, 3,
              (sliceAxis,
               sliceIndex,
               outputRegion
              )
            );

          if (isThresholdingOn)
          {
            bool useThresholdsWhenCalculatingEnclosedSeeds = false;

            this->DoThresholdApply(true, false, true);

            // Get seeds just on the current slice
            mitk::PointSet::Pointer seedsForCurrentSlice = mitk::PointSet::New();
            this->FilterSeedsToCurrentSlice(
                seeds,
                sliceAxis,
                sliceIndex,
                *(seedsForCurrentSlice.GetPointer())
                );

            // Reduce the list just down to those that are fully enclosed.
            mitk::PointSet::Pointer enclosedSeeds = mitk::PointSet::New();
            this->FilterSeedsToEnclosedSeedsOnCurrentSlice(
                *seedsForCurrentSlice,
                useThresholdsWhenCalculatingEnclosedSeeds,
                sliceIndex,
                *(enclosedSeeds.GetPointer())
                );

            // Do region growing, using only enclosed seeds.
            AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
                ITKUpdateRegionGrowing, 3,
                (false,
                 segmentationImage,
                 enclosedSeeds,
                 segmentationContours,
                 drawToolContours,
                 polyToolContours,
                 sliceAxis,
                 sliceIndex,
                 lowerThreshold,
                 upperThreshold,
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

            ITKCopyRegion<unsigned char, 3>(
                regionGrowingToItk->GetOutput(),
                sliceAxis,
                sliceIndex,
                outputToItk->GetOutput()
                );

            regionGrowingToItk = nullptr;
            outputToItk = nullptr;

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
                ITKUpdateRegionGrowing, 3,
                (false,
                 segmentationImage,
                 seeds,
                 segmentationContours,
                 drawToolContours,
                 polyToolContours,
                 sliceAxis,
                 sliceIndex,
                 referenceImage->GetStatistics()->GetScalarValueMinNoRecompute(),
                 referenceImage->GetStatistics()->GetScalarValueMaxNoRecompute(),
                 regionGrowingImage  // This is the image we are writing to.
                )
            );

          }

          // Then create filtered contours for the current slice.
          // So, if we are thresholding, we fit them round the current region growing image,
          // which if we have just used enclosed seeds above, will not include regions defined
          // by a seed and a threshold, but that have not been "applied" yet.

          AccessFixedDimensionByItk_n(referenceImage, // The reference image is the grey scale image (read only).
              ITKFilterContours, 3,
              (segmentationImage,
               seeds,
               *segmentationContours,
               *drawToolContours,
               *polyToolContours,
               sliceAxis,
               sliceIndex,
               lowerThreshold,
               upperThreshold,
               isThresholdingOn,
               *(copyOfInputContourSet.GetPointer()),
               *(outputContourSet.GetPointer())
              )
            );

          mitk::UndoStackItem::IncCurrObjectEventId();
          mitk::UndoStackItem::IncCurrGroupEventId();
          mitk::UndoStackItem::ExecuteIncrement();

          OpClean *doOp = new OpClean(OP_CLEAN, true, outputContourSet);
          OpClean *undoOp = new OpClean(OP_CLEAN, false, copyOfInputContourSet);
          mitk::OperationEvent* operationEvent = new mitk::OperationEvent(d->m_Interface, doOp, undoOp, "Clean: Filtering contours");
          mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
          this->ExecuteOperation(doOp);

          // Then we update the region growing to get up-to-date contours.
          this->UpdateRegionGrowing();

          if (!isThresholdingOn)
          {
            // Then we "apply" this region growing.
            OpThresholdApply::ProcessorPointer processor = OpThresholdApply::ProcessorType::New();
            OpThresholdApply *doApplyOp = new OpThresholdApply(OP_THRESHOLD_APPLY, true, outputRegion, processor, false);
            OpThresholdApply *undoApplyOp = new OpThresholdApply(OP_THRESHOLD_APPLY, false, outputRegion, processor, false);
            mitk::OperationEvent* operationApplyEvent = new mitk::OperationEvent(d->m_Interface, doApplyOp, undoApplyOp, "Clean: Calculate new image");
            mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationApplyEvent );
            this->ExecuteOperation(doApplyOp);

            // We should update the current slice contours, as the green contours
            // are the current segmentation that will be applied when we change slice.
            this->UpdateCurrentSliceContours();
          }

          drawTool->Clean(sliceIndex, sliceAxis);

          segmentationImage->Modified();
          segmentationNode->Modified();

          cleanWasPerformed = true;

        }
        catch(const mitk::AccessByItkException& e)
        {
          MITK_ERROR << "Could not do clean command: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        }
        catch(const itk::ExceptionObject& e)
        {
          MITK_ERROR << "Could not do clean command: Caught itk::ExceptionObject:" << e.what() << std::endl;
        }

        d->m_IsUpdating = wasUpdating;

      }
      else
      {
        MITK_ERROR << "Could not do clean operation: Error sliceAxis=" << sliceAxis << ", sliceIndex=" << sliceIndex << std::endl;
      }
    }
  }

  if (cleanWasPerformed)
  {
    this->RequestRenderWindowUpdate();
  }
}


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

void GeneralSegmentorController::ExecuteOperation(mitk::Operation* operation)
{
  Q_D(GeneralSegmentorController);

  if (!this->HasInitialisedWorkingData())
  {
    return;
  }

  if (!operation)
  {
    return;
  }

  mitk::Image::Pointer segmentationImage = this->GetWorkingImage(MIDASTool::SEGMENTATION);
  assert(segmentationImage);

  mitk::DataNode::Pointer segmentationNode = this->GetWorkingData()[MIDASTool::SEGMENTATION];
  assert(segmentationNode);

  mitk::Image* referenceImage = this->GetReferenceImage();
  assert(referenceImage);

  mitk::Image* regionGrowingImage = this->GetWorkingImage(MIDASTool::REGION_GROWING);
  assert(regionGrowingImage);

  mitk::PointSet* seeds = this->GetSeeds();
  assert(seeds);

  mitk::DataNode::Pointer seedsNode = this->GetWorkingData()[MIDASTool::SEEDS];
  assert(seedsNode);

  switch (operation->GetOperationType())
  {
  case OP_CHANGE_SLICE:
    {
      // Simply to make sure we can switch slice, and undo/redo it.
      OpChangeSliceCommand* op = dynamic_cast<OpChangeSliceCommand*>(operation);
      assert(op);

      mitk::Point3D beforePoint = op->GetBeforePoint();
      mitk::Point3D afterPoint = op->GetAfterPoint();

      mitk::Point3D selectedPoint;

      if (op->IsRedo())
      {
        selectedPoint = afterPoint;
      }
      else
      {
        selectedPoint = beforePoint;
      }

      bool wasChangingSlice = d->m_IsChangingSlice;
      d->m_IsChangingSlice = true;
      this->GetView()->SetSelectedPosition(selectedPoint);
      d->m_IsChangingSlice = wasChangingSlice;

      break;
    }
  case OP_PROPAGATE_SEEDS:
    {
      OpPropagateSeeds* op = dynamic_cast<OpPropagateSeeds*>(operation);
      assert(op);

      mitk::PointSet* newSeeds = op->GetSeeds();
      assert(newSeeds);

      mitk::CopyPointSets(*newSeeds, *seeds);

      seeds->Modified();
      seedsNode->Modified();

      break;
    }
  case OP_RETAIN_MARKS:
    {
      try
      {
        OpRetainMarks* op = dynamic_cast<OpRetainMarks*>(operation);
        assert(op);

        OpRetainMarks::ProcessorType::Pointer processor = op->GetProcessor();
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

        targetImageToItk = nullptr;

        mitk::Image::Pointer outputImage = mitk::ImportItkImage( processor->GetDestinationImage());

        processor->SetSourceImage(nullptr);
        processor->SetDestinationImage(nullptr);

        segmentationNode->SetData(outputImage);
        segmentationNode->Modified();
      }
      catch(const itk::ExceptionObject& e)
      {
        MITK_ERROR << "Could not do retain marks: Caught itk::ExceptionObject:" << e.what() << std::endl;
        return;
      }

      break;
    }
  case OP_THRESHOLD_APPLY:
    {
      OpThresholdApply *op = dynamic_cast<OpThresholdApply*>(operation);
      assert(op);

      try
      {
        AccessFixedDimensionByItk_n(referenceImage, ITKPropagateToSegmentationImage, 3,
              (
                segmentationImage,
                regionGrowingImage,
                op
              )
            );

        d->m_GUI->SetThresholdingCheckBoxChecked(op->GetThresholdFlag());
        d->m_GUI->SetThresholdingWidgetsEnabled(op->GetThresholdFlag());

        segmentationImage->Modified();
        segmentationNode->Modified();

        regionGrowingImage->Modified();

      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Could not do threshold: Caught mitk::AccessByItkException:" << e.what() << std::endl;
        return;
      }
      catch(const itk::ExceptionObject& e)
      {
        MITK_ERROR << "Could not do threshold: Caught itk::ExceptionObject:" << e.what() << std::endl;
        return;
      }

      break;
    }
  case OP_CLEAN:
    {
      try
      {
        OpClean* op = dynamic_cast<OpClean*>(operation);
        assert(op);

        mitk::ContourModelSet* newContours = op->GetContourSet();
        assert(newContours);

        mitk::ContourModelSet* contoursToReplace = dynamic_cast<mitk::ContourModelSet*>(this->GetWorkingData()[MIDASTool::CONTOURS]->GetData());
        assert(contoursToReplace);

        MIDASContourTool::CopyContourSet(*newContours, *contoursToReplace);
        contoursToReplace->Modified();
        this->GetWorkingData()[MIDASTool::CONTOURS]->Modified();

        segmentationImage->Modified();
        segmentationNode->Modified();

      }
      catch(const itk::ExceptionObject& e)
      {
        MITK_ERROR << "Could not do clean: Caught itk::ExceptionObject:" << e.what() << std::endl;
        return;
      }

      break;
    }
  case OP_WIPE:
    {
      OpWipe *op = dynamic_cast<OpWipe*>(operation);
      assert(op);

      try
      {
        AccessFixedTypeByItk_n(segmentationImage,
            ITKDoWipe,
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
      catch(const itk::ExceptionObject& e)
      {
        MITK_ERROR << "Could not do wipe: Caught itk::ExceptionObject:" << e.what() << std::endl;
        return;
      }

      break;
    }
  case OP_PROPAGATE:
    {
      OpPropagate *op = dynamic_cast<OpPropagate*>(operation);
      assert(op);

      try
      {
        AccessFixedDimensionByItk_n(referenceImage, ITKPropagateToSegmentationImage, 3,
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
      catch(const itk::ExceptionObject& e)
      {
        MITK_ERROR << "Could not do propagation: Caught itk::ExceptionObject:" << e.what() << std::endl;
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

}
