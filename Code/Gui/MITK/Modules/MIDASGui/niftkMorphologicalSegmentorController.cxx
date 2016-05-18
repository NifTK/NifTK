/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMorphologicalSegmentorController.h"

#include <QMessageBox>

#include <mitkImageStatisticsHolder.h>
#include <mitkITKRegionParametersDataNodeProperty.h>
#include <mitkPlane.h>
#include <mitkUndoController.h>

#include <mitkDataStorageUtils.h>
#include <niftkIBaseView.h>
#include <niftkMIDASPaintbrushTool.h>

#include "Internal/niftkMorphologicalSegmentorGUI.h"

namespace niftk
{

//-----------------------------------------------------------------------------
MorphologicalSegmentorController::MorphologicalSegmentorController(IBaseView* view)
  : BaseSegmentorController(view),
    m_MorphologicalSegmentorGUI(nullptr),
    m_PipelineManager(nullptr),
    m_TabIndex(-1)
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  toolManager->RegisterTool("MIDASPaintbrushTool");

  MIDASPaintbrushTool* paintbrushTool = this->GetToolByType<MIDASPaintbrushTool>();
  assert(paintbrushTool);

  paintbrushTool->InstallEventFilter(this);

  paintbrushTool->SegmentationEdited.AddListener(mitk::MessageDelegate1<MorphologicalSegmentorController, int>(this, &MorphologicalSegmentorController::OnSegmentationEdited));

  m_PipelineManager = MorphologicalSegmentorPipelineManager::New();
  m_PipelineManager->SetDataStorage(this->GetDataStorage());
  m_PipelineManager->SetToolManager(this->GetToolManager());
}


//-----------------------------------------------------------------------------
MorphologicalSegmentorController::~MorphologicalSegmentorController()
{
  MIDASPaintbrushTool* paintbrushTool = this->GetToolByType<MIDASPaintbrushTool>();
  assert(paintbrushTool);

  paintbrushTool->SegmentationEdited.RemoveListener(mitk::MessageDelegate1<MorphologicalSegmentorController, int>(this, &MorphologicalSegmentorController::OnSegmentationEdited));

  paintbrushTool->RemoveEventFilter(this);
}


//-----------------------------------------------------------------------------
BaseGUI* MorphologicalSegmentorController::CreateGUI(QWidget* parent)
{
  return new MorphologicalSegmentorGUI(parent);
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::SetupGUI(QWidget* parent)
{
  BaseSegmentorController::SetupGUI(parent);

  m_MorphologicalSegmentorGUI = dynamic_cast<MorphologicalSegmentorGUI*>(this->GetSegmentorGUI());

  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(ThresholdingValuesChanged(double, double, int)), SLOT(OnThresholdingValuesChanged(double, double, int)));
  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(ErosionsValuesChanged(double, int)), SLOT(OnErosionsValuesChanged(double, int)));
  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(DilationsValuesChanged(double, double, int)), SLOT(OnDilationsValuesChanged(double, double, int)));
  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(RethresholdingValuesChanged(int)), SLOT(OnRethresholdingValuesChanged(int)));
  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(TabChanged(int)), SLOT(OnTabChanged(int)));
  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(OKButtonClicked()), SLOT(OnOKButtonClicked()));
//  this->connect(m_MorphologicalControls, SIGNAL(CancelButtonClicked()), SLOT(OnCancelButtonClicked()));
  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(RestartButtonClicked()), SLOT(OnRestartButtonClicked()));
}


//-----------------------------------------------------------------------------
bool MorphologicalSegmentorController::IsASegmentationImage(const mitk::DataNode::Pointer node)
{
  return m_PipelineManager->IsNodeASegmentationImage(node);
}


//-----------------------------------------------------------------------------
bool MorphologicalSegmentorController::IsAWorkingImage(const mitk::DataNode::Pointer node)
{
  return m_PipelineManager->IsNodeAWorkingImage(node);
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType MorphologicalSegmentorController::GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  return m_PipelineManager->GetWorkingDataFromSegmentationNode(node);
}


//-----------------------------------------------------------------------------
mitk::DataNode* MorphologicalSegmentorController::GetSegmentationNodeFromWorkingData(const mitk::DataNode::Pointer node)
{
  return m_PipelineManager->GetSegmentationNodeFromWorkingData(node);
}


//-----------------------------------------------------------------------------
bool MorphologicalSegmentorController::CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node)
{
  return m_PipelineManager->CanStartSegmentationForBinaryNode(node);
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnNewSegmentationButtonClicked()
{
  BaseSegmentorController::OnNewSegmentationButtonClicked();

  // Create the new segmentation, either using a previously selected one, or create a new volume.
  mitk::DataNode::Pointer newSegmentation = NULL;
  bool isRestarting = false;

  // Make sure we have a reference images... which should always be true at this point.
  mitk::Image::ConstPointer image = m_PipelineManager->GetReferenceImage();
  if (image.IsNotNull())
  {

    // Make sure we can retrieve the paintbrush tool, which can be used to create a new segmentation image.
    mitk::ToolManager* toolManager = this->GetToolManager();
    assert(toolManager);

    int paintbrushToolId = toolManager->GetToolIdByToolType<MIDASPaintbrushTool>();

    mitk::Tool* paintbrushTool = toolManager->GetToolById(paintbrushToolId);
    assert(paintbrushTool);

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

      // The above method returns NULL if the user exited the colour selection dialog box.
      if (newSegmentation.IsNull())
      {
        return;
      }
    }

    mitk::DataNode::Pointer axialCutOffPlaneNode = this->CreateAxialCutOffPlaneNode(image);
    this->GetDataStorage()->Add(axialCutOffPlaneNode, newSegmentation);

    this->WaitCursorOn();

    // Mark the newSegmentation as "unfinished".
    newSegmentation->SetBoolProperty(MorphologicalSegmentorPipelineManager::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED.c_str(), false);

    try
    {
      // Create that orange colour that MIDAS uses to highlight edited regions.
      mitk::ColorProperty::Pointer col = mitk::ColorProperty::New();
      col->SetColor((float)1.0, (float)(165.0/255.0), (float)0.0);

      // Create additions data node, and store reference to image
      float segCol[3];
      newSegmentation->GetColor(segCol);
      mitk::ColorProperty::Pointer segmentationColor = mitk::ColorProperty::New(segCol[0], segCol[1], segCol[2]);


      // Create extra data and store with ToolManager
      mitk::ITKRegionParametersDataNodeProperty::Pointer erodeAddEditingProp = mitk::ITKRegionParametersDataNodeProperty::New();
      erodeAddEditingProp->SetSize(1,1,1);
      erodeAddEditingProp->SetValid(false);
      mitk::DataNode::Pointer erodeAddNode = paintbrushTool->CreateEmptySegmentationNode(image, MIDASPaintbrushTool::EROSIONS_ADDITIONS_NAME, col->GetColor());
      erodeAddNode->SetBoolProperty("helper object", true);
      erodeAddNode->SetBoolProperty("visible", false);
      erodeAddNode->SetColor(segCol);
      erodeAddNode->SetProperty("binaryimage.selectedcolor", segmentationColor);
      erodeAddNode->AddProperty(MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str(), erodeAddEditingProp);

      mitk::ITKRegionParametersDataNodeProperty::Pointer erodeSubtractEditingProp = mitk::ITKRegionParametersDataNodeProperty::New();
      erodeSubtractEditingProp->SetSize(1,1,1);
      erodeSubtractEditingProp->SetValid(false);
      mitk::DataNode::Pointer erodeSubtractNode = paintbrushTool->CreateEmptySegmentationNode( image, MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS_NAME, col->GetColor());
      erodeSubtractNode->SetBoolProperty("helper object", true);
      erodeSubtractNode->SetBoolProperty("visible", false);
      erodeSubtractNode->SetColor(col->GetColor());
      erodeSubtractNode->SetProperty("binaryimage.selectedcolor", col);
      erodeSubtractNode->AddProperty(MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str(), erodeSubtractEditingProp);

      mitk::ITKRegionParametersDataNodeProperty::Pointer dilateAddEditingProp = mitk::ITKRegionParametersDataNodeProperty::New();
      dilateAddEditingProp->SetSize(1,1,1);
      dilateAddEditingProp->SetValid(false);
      mitk::DataNode::Pointer dilateAddNode = paintbrushTool->CreateEmptySegmentationNode( image, MIDASPaintbrushTool::DILATIONS_ADDITIONS_NAME, col->GetColor());
      dilateAddNode->SetBoolProperty("helper object", true);
      dilateAddNode->SetBoolProperty("visible", false);
      dilateAddNode->SetColor(segCol);
      dilateAddNode->SetProperty("binaryimage.selectedcolor", segmentationColor);
      dilateAddNode->AddProperty(MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str(), dilateAddEditingProp);

      mitk::ITKRegionParametersDataNodeProperty::Pointer dilateSubtractEditingProp = mitk::ITKRegionParametersDataNodeProperty::New();
      dilateSubtractEditingProp->SetSize(1,1,1);
      dilateSubtractEditingProp->SetValid(false);
      mitk::DataNode::Pointer dilateSubtractNode = paintbrushTool->CreateEmptySegmentationNode( image, MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS_NAME, col->GetColor());
      dilateSubtractNode->SetBoolProperty("helper object", true);
      dilateSubtractNode->SetBoolProperty("visible", false);
      dilateSubtractNode->SetColor(col->GetColor());
      dilateSubtractNode->SetProperty("binaryimage.selectedcolor", col);
      dilateSubtractNode->AddProperty(MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str(), dilateSubtractEditingProp);

      this->ApplyDisplayOptions(erodeAddNode);
      this->ApplyDisplayOptions(erodeSubtractNode);
      this->ApplyDisplayOptions(dilateAddNode);
      this->ApplyDisplayOptions(dilateSubtractNode);

      // Add the image to data storage, and specify this derived image as the one the toolManager will edit to.
      this->GetDataStorage()->Add(erodeAddNode, newSegmentation); // add as a child, because the segmentation "derives" from the original
      this->GetDataStorage()->Add(erodeSubtractNode, newSegmentation); // add as a child, because the segmentation "derives" from the original
      this->GetDataStorage()->Add(dilateAddNode, newSegmentation); // add as a child, because the segmentation "derives" from the original
      this->GetDataStorage()->Add(dilateSubtractNode, newSegmentation); // add as a child, because the segmentation "derives" from the original

      // Set working data. Compare with MIDASGeneralSegmentorView.
      // Note the order:
      //
      // 1. The First image is the "Additions" image for erosions, that we can manually add data/voxels to.
      // 2. The Second image is the "Subtractions" image for erosions, that is used for connection breaker.
      // 3. The Third image is the "Additions" image for dilations, that we can manually add data/voxels to.
      // 4. The Forth image is the "Subtractions" image for dilations, that is used for connection breaker.
      //
      // This must match the order in:
      //
      // 1. niftkMorphologicalSegmentorPipelineManager::UpdateSegmentation()
      // 2. niftkMIDASPaintbrushTool.
      // and unit tests etc. Probably best to search for
      // MORPH_EDITS_EROSIONS_SUBTRACTIONS
      // MORPH_EDITS_EROSIONS_ADDITIONS
      // MORPH_EDITS_DILATIONS_SUBTRACTIONS
      // MORPH_EDITS_DILATIONS_ADDITIONS

      mitk::ToolManager::DataVectorType workingData(4);
      workingData[MIDASPaintbrushTool::EROSIONS_ADDITIONS] = erodeAddNode;
      workingData[MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS] = erodeSubtractNode;
      workingData[MIDASPaintbrushTool::DILATIONS_ADDITIONS] = dilateAddNode;
      workingData[MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS] = dilateSubtractNode;

      toolManager->SetWorkingData(workingData);

      // Set properties, and then the control values to match.
      if (isRestarting)
      {
        newSegmentation->SetBoolProperty("midas.morph.restarting", true);
        this->SetControlsFromSegmentationNodeProps();
        m_PipelineManager->UpdateSegmentation();
      }
      else
      {
        this->SetSegmentationNodePropsFromReferenceImage();
        this->SetControlsFromReferenceImage();
        this->SetControlsFromSegmentationNodeProps();
        m_PipelineManager->UpdateSegmentation();
      }
    }
    catch (std::bad_alloc&)
    {
      QMessageBox::warning(NULL,"Create new segmentation","Could not allocate memory for new segmentation");
    }

    this->GetView()->FocusOnCurrentWindow();
    this->RequestRenderWindowUpdate();
    this->WaitCursorOff();

  } // end if we have a reference image

  // Finally, select the new segmentation node.
  this->GetView()->SetCurrentSelection(newSegmentation);
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnDataManagerSelectionChanged(const QList<mitk::DataNode::Pointer>& nodes)
{
  BaseSegmentorController::OnDataManagerSelectionChanged(nodes);

  bool enableWidgets = false;

  if (nodes.size() == 1)
  {
    mitk::Image::ConstPointer referenceImage = m_PipelineManager->GetReferenceImage();
    mitk::Image::Pointer segmentationImage = m_PipelineManager->GetSegmentationImage();

    if (referenceImage.IsNotNull() && segmentationImage.IsNotNull())
    {
      this->SetControlsFromSegmentationNodeProps();
    }

    bool isAlreadyFinished = true;
    bool foundAlreadyFinishedProperty = nodes[0]->GetBoolProperty(MorphologicalSegmentorPipelineManager::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED.c_str(), isAlreadyFinished);

    if (foundAlreadyFinishedProperty && !isAlreadyFinished)
    {
      enableWidgets = true;
    }
  }

  m_MorphologicalSegmentorGUI->EnableSegmentationWidgets(enableWidgets);
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnThresholdingValuesChanged(double lowerThreshold, double upperThreshold, int axialSliceNumber)
{
  m_PipelineManager->OnThresholdingValuesChanged(lowerThreshold, upperThreshold, axialSliceNumber);

  mitk::DataNode::Pointer referenceImageNode = this->GetReferenceNode();
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNode();
  mitk::Image* referenceImage = dynamic_cast<mitk::Image*>(referenceImageNode->GetData());
  mitk::BaseGeometry* geometry = referenceImage->GetGeometry();

  int axialAxis = niftk::GetThroughPlaneAxis(referenceImage, IMAGE_ORIENTATION_AXIAL);
  int axialUpDirection = niftk::GetUpDirection(referenceImage, IMAGE_ORIENTATION_AXIAL);

  mitk::Plane* axialCutOffPlane = this->GetDataStorage()->GetNamedDerivedObject<mitk::Plane>("Axial cut-off plane", segmentationNode);

  // Lift the axial cut-off plane to the height determined by axialSliceNumber.
  mitk::Point3D planeCentre = axialCutOffPlane->GetGeometry()->GetOrigin();
  planeCentre[2] = geometry->GetOrigin()[2] + (axialUpDirection * axialSliceNumber - 0.5) * geometry->GetSpacing()[axialAxis];
  axialCutOffPlane->SetOrigin(planeCentre);

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnErosionsValuesChanged(double upperThreshold, int numberOfErosions)
{
  m_PipelineManager->OnErosionsValuesChanged(upperThreshold, numberOfErosions);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnDilationsValuesChanged(double lowerPercentage, double upperPercentage, int numberOfDilations)
{
  m_PipelineManager->OnDilationsValuesChanged(lowerPercentage, upperPercentage, numberOfDilations);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnRethresholdingValuesChanged(int boxSize)
{
  m_PipelineManager->OnRethresholdingValuesChanged(boxSize);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnTabChanged(int tabIndex)
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNode();
  if (segmentationNode.IsNotNull())
  {
    if (tabIndex == 1 || tabIndex == 2)
    {
      m_MorphologicalSegmentorGUI->SetToolSelectorEnabled(true);

      mitk::ToolManager::Pointer toolManager = this->GetToolManager();
      MIDASPaintbrushTool::Pointer paintbrushTool = this->GetToolByType<MIDASPaintbrushTool>();

      mitk::DataNode::Pointer erodeSubtractNode = this->GetToolManager()->GetWorkingData(MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS);
      mitk::DataNode::Pointer dilateSubtractNode = this->GetToolManager()->GetWorkingData(MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS);

      if (tabIndex == 1)
      {
        paintbrushTool->SetErosionMode(true);
        erodeSubtractNode->SetVisibility(true);
        dilateSubtractNode->SetVisibility(false);

        // Only if we are switching from tab 2 to 1.
        if (m_TabIndex == 2)
        {
          const mitk::Image* dilateSubtractImage = dynamic_cast<mitk::Image*>(dilateSubtractNode->GetData());
          mitk::Image* erodeSubtractImage = dynamic_cast<mitk::Image*>(erodeSubtractNode->GetData());
          if (dilateSubtractImage != NULL && erodeSubtractImage != NULL)
          {
//            m_PipelineManager->SetErosionSubtractionsInput(0);
//            mitk::CopyIntensityData(dilateSubtractImage, erodeSubtractImage);
//            m_PipelineManager->SetErosionSubtractionsInput(erodeSubtractImage);
          }
        }
      }
      else // i==2
      {
        paintbrushTool->SetErosionMode(false);
        erodeSubtractNode->SetVisibility(false);
        dilateSubtractNode->SetVisibility(true);

        // Only if we are switching from tab 1 to 2.
        if (m_TabIndex == 1)
        {
          const mitk::Image* erodeSubtractImage = dynamic_cast<mitk::Image*>(erodeSubtractNode->GetData());
          mitk::Image* dilateSubtractImage = dynamic_cast<mitk::Image*>(dilateSubtractNode->GetData());
          if (erodeSubtractImage != NULL && dilateSubtractImage != NULL)
          {
//            m_PipelineManager->SetDilationSubtractionsInput(0);
//            mitk::CopyIntensityData(erodeSubtractImage, dilateSubtractImage);
//            m_PipelineManager->SetDilationSubtractionsInput(dilateSubtractImage);
          }
        }
      }
    }
    else
    {
      m_MorphologicalSegmentorGUI->SetToolSelectorEnabled(false);
      this->OnToolSelected(-1); // make sure we de-activate tools.
    }

    m_PipelineManager->OnTabChanged(tabIndex);

    this->RequestRenderWindowUpdate();
  }

  m_TabIndex = tabIndex;
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnOKButtonClicked()
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNode();
  if (segmentationNode.IsNotNull())
  {
    this->OnToolSelected(-1);
    m_MorphologicalSegmentorGUI->EnableSegmentationWidgets(false);
    m_MorphologicalSegmentorGUI->SetTabIndex(0);
    m_PipelineManager->FinalizeSegmentation();

    /// Remove the axial cut-off plane node from the data storage.
    mitk::DataNode::Pointer axialCutOffPlaneNode = this->GetDataStorage()->GetNamedDerivedNode("Axial cut-off plane", segmentationNode);
    this->GetDataStorage()->Remove(axialCutOffPlaneNode);

    this->GetView()->FireNodeSelected(this->GetReferenceNode());
    this->RequestRenderWindowUpdate();
    mitk::UndoController::GetCurrentUndoModel()->Clear();
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnRestartButtonClicked()
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNode();
  if (segmentationNode.IsNotNull())
  {
    this->OnToolSelected(-1);
    m_PipelineManager->ClearWorkingData();
    this->SetSegmentationNodePropsFromReferenceImage();
    this->SetControlsFromReferenceImage();
    this->SetControlsFromSegmentationNodeProps();
    m_PipelineManager->UpdateSegmentation();

    /// Reset the axial cut-off plane to the bottom of the image.
    {
      mitk::DataNode::Pointer referenceImageNode = this->GetReferenceNode();
      mitk::Image* referenceImage = dynamic_cast<mitk::Image*>(referenceImageNode->GetData());
      mitk::BaseGeometry* geometry = referenceImage->GetGeometry();

      mitk::Plane* axialCutOffPlane = this->GetDataStorage()->GetNamedDerivedObject<mitk::Plane>("Axial cut-off plane", segmentationNode);

      int axialAxis = GetThroughPlaneAxis(referenceImage, IMAGE_ORIENTATION_AXIAL);

      // The centre of the plane is the same as the centre of the image, but it is shifted
      // along the axial axis to a position determined by axialSliceNumber.
      // As an initial point we set it one slice below the 'height' of the origin.
      mitk::Point3D planeCentre = geometry->GetCenter();
      planeCentre[2] = geometry->GetOrigin()[axialAxis] - geometry->GetSpacing()[axialAxis];

      axialCutOffPlane->SetOrigin(planeCentre);
    }

    this->GetView()->FireNodeSelected(segmentationNode);
    this->RequestRenderWindowUpdate();
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnCancelButtonClicked()
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNode();
  if (segmentationNode.IsNotNull())
  {
    this->OnToolSelected(-1);
    m_MorphologicalSegmentorGUI->EnableSegmentationWidgets(false);
    m_MorphologicalSegmentorGUI->SetTabIndex(0);
    m_PipelineManager->RemoveWorkingData();
    mitk::Image::Pointer segmentationImage = dynamic_cast<mitk::Image*>(segmentationNode->GetData());
    m_PipelineManager->DestroyPipeline(segmentationImage);
    mitk::DataNode::Pointer axialCutOffPlaneNode = this->GetDataStorage()->GetNamedDerivedNode("Axial cut-off plane", segmentationNode);
    this->GetDataStorage()->Remove(axialCutOffPlaneNode);
    this->GetDataStorage()->Remove(segmentationNode);
    this->GetView()->FireNodeSelected(this->GetReferenceNode());
    this->RequestRenderWindowUpdate();
    mitk::UndoController::GetCurrentUndoModel()->Clear();
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnViewGetsClosed()
{
  /// TODO this is not invoked at all.
  /// This function was called "ClosePart" before it was moved here from niftkMorphologicalSegmentorView.
  /// It was not invoked there, either. I leave this here to remind me that the segmentation should
  /// be discarded when the view is closed.
  if  (m_PipelineManager->HasSegmentationNode())
  {
    this->OnCancelButtonClicked();
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnSegmentationEdited(int imageIndex)
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  if (toolManager)
  {
    mitk::DataNode* node = toolManager->GetWorkingData(imageIndex);
    assert(node);
    mitk::ITKRegionParametersDataNodeProperty::Pointer prop =
        dynamic_cast<mitk::ITKRegionParametersDataNodeProperty*>(node->GetProperty(MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str()));
    if (prop.IsNotNull() && prop->HasVolume())
    {
      m_PipelineManager->UpdateSegmentation();
    }
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnNodeRemoved(const mitk::DataNode* removedNode)
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNode();
  if (segmentationNode.IsNotNull() && segmentationNode.GetPointer() == removedNode)
  {
    this->OnToolSelected(-1);
    m_MorphologicalSegmentorGUI->EnableSegmentationWidgets(false);
    m_MorphologicalSegmentorGUI->SetTabIndex(0);
    m_PipelineManager->RemoveWorkingData();
    mitk::Image::Pointer segmentationImage = dynamic_cast<mitk::Image*>(segmentationNode->GetData());
    m_PipelineManager->DestroyPipeline(segmentationImage);
//    this->GetDataStorage()->Remove(segmentationNode);
    this->GetView()->FireNodeSelected(this->GetReferenceNode());
    this->RequestRenderWindowUpdate();
    mitk::UndoController::GetCurrentUndoModel()->Clear();
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnNodeVisibilityChanged(const mitk::DataNode* node, const mitk::BaseRenderer* /*renderer*/)
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNode();

  std::vector<mitk::DataNode*> workingData = this->GetWorkingData();
  if (segmentationNode.IsNotNull() && node == segmentationNode && workingData.size() == 4)
  {
    mitk::DataNode::Pointer axialCutOffPlaneNode = this->GetDataStorage()->GetNamedDerivedNode("Axial cut-off plane", segmentationNode);

    bool segmentationNodeVisibility;
    if (node->GetVisibility(segmentationNodeVisibility, 0) && segmentationNodeVisibility)
    {
      int tabIndex = m_MorphologicalSegmentorGUI->GetTabIndex();
      workingData[MIDASPaintbrushTool::EROSIONS_ADDITIONS]->SetVisibility(false);
      workingData[MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS]->SetVisibility(tabIndex == 1);
      workingData[MIDASPaintbrushTool::DILATIONS_ADDITIONS]->SetVisibility(false);
      workingData[MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS]->SetVisibility(tabIndex == 2);
      axialCutOffPlaneNode->SetVisibility(true);
    }
    else
    {
      for (std::size_t i = 1; i < workingData.size(); ++i)
      {
        workingData[i]->SetVisibility(false);
      }
      axialCutOffPlaneNode->SetVisibility(false);
    }
  }
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer MorphologicalSegmentorController::CreateAxialCutOffPlaneNode(const mitk::Image* referenceImage)
{
  mitk::BaseGeometry* geometry = referenceImage->GetGeometry();

  int axialAxis = niftk::GetThroughPlaneAxis(referenceImage, IMAGE_ORIENTATION_AXIAL);
  int sagittalAxis = niftk::GetThroughPlaneAxis(referenceImage, IMAGE_ORIENTATION_SAGITTAL);
  int coronalAxis = niftk::GetThroughPlaneAxis(referenceImage, IMAGE_ORIENTATION_CORONAL);

  int axialUpDirection = niftk::GetUpDirection(referenceImage, IMAGE_ORIENTATION_AXIAL);
  int sagittalUpDirection = niftk::GetUpDirection(referenceImage, IMAGE_ORIENTATION_SAGITTAL);
  int coronalUpDirection = niftk::GetUpDirection(referenceImage, IMAGE_ORIENTATION_CORONAL);

  /// The centre of the plane is the same as the centre of the image, but it is shifted
  /// along the axial axis to a position determined by axialSliceNumber.
  /// As an initial point we set it one slice below the 'height' of the origin.
  /// The world coordinate always increases from the bottom to the top, but the slice
  /// numbering depends on the image. (This is what the 'up direction' tells.)
  mitk::Point3D planeCentre = geometry->GetCenter();
  mitk::Vector3D spacing = geometry->GetSpacing();
  planeCentre[0] += sagittalUpDirection * 0.5 * spacing[sagittalAxis];
  planeCentre[1] += coronalUpDirection * 0.5 * spacing[coronalAxis];
  planeCentre[2] = geometry->GetOrigin()[2] - axialUpDirection * 0.5 * spacing[axialAxis];
  if (axialUpDirection == -1)
  {
    planeCentre[2] -= geometry->GetExtentInMM(axialAxis);
  }

  mitk::Plane::Pointer axialCutOffPlane = mitk::Plane::New();
  axialCutOffPlane->SetOrigin(planeCentre);

  /// The size of the plane is the size of the image in the other two directions.
  axialCutOffPlane->SetExtent(geometry->GetExtentInMM(sagittalAxis), geometry->GetExtentInMM(coronalAxis));

  mitk::DataNode::Pointer axialCutOffPlaneNode = mitk::DataNode::New();
  axialCutOffPlaneNode->SetName("Axial cut-off plane");
  axialCutOffPlaneNode->SetColor(1.0, 1.0, 0.0);
  axialCutOffPlaneNode->SetIntProperty("layer", 1000);
  axialCutOffPlaneNode->SetOpacity(0.5);
  axialCutOffPlaneNode->SetBoolProperty("helper object", true);
  axialCutOffPlaneNode->SetBoolProperty("includeInBoundingBox", false);

  axialCutOffPlaneNode->SetVisibility(true);

  // This is for the DnD display, so that it does not try to change the
  // visibility after node addition.
  axialCutOffPlaneNode->SetBoolProperty("managed visibility", false);

  // Put the data into the node.
  axialCutOffPlaneNode->SetData(axialCutOffPlane);

  return axialCutOffPlaneNode;
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::SetSegmentationNodePropsFromReferenceImage()
{
  m_PipelineManager->SetSegmentationNodePropsFromReferenceImage();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::SetControlsFromReferenceImage()
{
  mitk::Image::ConstPointer referenceImage = m_PipelineManager->GetReferenceImage();
  if (referenceImage.IsNotNull())
  {
    int axialAxis = this->GetReferenceImageSliceAxis(IMAGE_ORIENTATION_AXIAL);
    int numberOfAxialSlices = referenceImage->GetDimension(axialAxis);
    int upDirection = niftk::GetUpDirection(referenceImage, IMAGE_ORIENTATION_AXIAL);

    m_MorphologicalSegmentorGUI->SetControlsByReferenceImage(
        referenceImage->GetStatistics()->GetScalarValueMin(),
        referenceImage->GetStatistics()->GetScalarValueMax(),
        numberOfAxialSlices,
        upDirection);
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::SetControlsFromSegmentationNodeProps()
{
  MorphologicalSegmentorPipelineParams params;
  m_PipelineManager->GetPipelineParamsFromSegmentationNode(params);

  m_MorphologicalSegmentorGUI->SetControlsByPipelineParams(params);
}

}
