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

#include <mitkImageAccessByItk.h>
#include <mitkImageStatisticsHolder.h>
#include <mitkPlane.h>
#include <mitkUndoController.h>

#include <niftkDataStorageUtils.h>
#include <niftkIBaseView.h>
#include <niftkITKRegionParametersDataNodeProperty.h>
#include <niftkPaintbrushTool.h>

#include "Internal/niftkMorphologicalSegmentorGUI.h"

/// Two utility functions are used from here to check if an image is empty and to clear it.
/// They are not specific to the general segmentor.
#include <niftkGeneralSegmentorUtils.h>

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
  toolManager->RegisterTool("PaintbrushTool");

  PaintbrushTool* paintbrushTool = this->GetToolByType<PaintbrushTool>();
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
  PaintbrushTool* paintbrushTool = this->GetToolByType<PaintbrushTool>();
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
  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(CancelButtonClicked()), SLOT(OnCancelButtonClicked()));
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
std::vector<mitk::DataNode*> MorphologicalSegmentorController::GetWorkingNodesFromSegmentationNode(const mitk::DataNode::Pointer segmentationNode)
{
  assert(segmentationNode.IsNotNull());

  std::vector<mitk::DataNode*> workingNodes(5);
  std::fill(workingNodes.begin(), workingNodes.end(), (mitk::DataNode*) 0);

  workingNodes[PaintbrushTool::SEGMENTATION] = segmentationNode;

  mitk::DataStorage::SetOfObjects::Pointer derivedNodes = niftk::FindDerivedImages(this->GetDataStorage(), segmentationNode, true );

  for (std::size_t i = 0; i < derivedNodes->size(); i++)
  {
    mitk::DataNode::Pointer derivedNode = derivedNodes->at(i);
    std::string name = derivedNode->GetName();
    if (name == PaintbrushTool::EROSIONS_ADDITIONS_NAME)
    {
      workingNodes[PaintbrushTool::EROSIONS_ADDITIONS] = derivedNode;
    }
    else if (name == PaintbrushTool::EROSIONS_SUBTRACTIONS_NAME)
    {
      workingNodes[PaintbrushTool::EROSIONS_SUBTRACTIONS] = derivedNode;
    }
    else if (name == PaintbrushTool::DILATIONS_ADDITIONS_NAME)
    {
      workingNodes[PaintbrushTool::DILATIONS_ADDITIONS] = derivedNode;
    }
    else if (name == PaintbrushTool::DILATIONS_SUBTRACTIONS_NAME)
    {
      workingNodes[PaintbrushTool::DILATIONS_SUBTRACTIONS] = derivedNode;
    }
  }

  if (std::count(workingNodes.begin(), workingNodes.end(), (mitk::DataNode*) 0) != 0)
  {
    MITK_INFO << "Working data nodes missing for the morphological segmentation pipeline.";
    workingNodes.clear();
  }

  return workingNodes;
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnNewSegmentationButtonClicked()
{
  /// Note:
  /// The 'new segmentation' button is enabled only when a reference image is selected.
  /// A reference image gets selected when the selection in the data manager changes to a valid
  /// reference image or a segmentation that was created by this segmentor.
  /// Hence, we can assume that we have a valid tool manager, paintbrush tool and reference image.

  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  const mitk::Image* referenceImage = this->GetReferenceImage();
  assert(referenceImage);

  mitk::Tool* paintbrushTool = this->GetToolByType<PaintbrushTool>();
  assert(paintbrushTool);

  QList<mitk::DataNode::Pointer> selectedNodes = this->GetView()->GetDataManagerSelection();
  if (selectedNodes.size() != 1)
  {
    return;
  }

  mitk::DataNode::Pointer selectedNode = selectedNodes.at(0);

  /// Create the new segmentation, either using a previously selected one, or create a new volume.
  mitk::DataNode::Pointer newSegmentation;
  bool isRestarting = false;

  if (niftk::IsNodeABinaryImage(selectedNode)
      && this->CanStartSegmentationForBinaryNode(selectedNode)
      && !this->IsASegmentationImage(selectedNode)
      )
  {
    newSegmentation =  selectedNode;
    isRestarting = true;

    if (!newSegmentation->GetProperty("midas.morph.stage"))
    {
      /// The segmentation is started on an already existing binary image, but the pipeline
      /// has not been performed on this data. The image contents will need to be erased.
      bool imageIsEmpty;
      const mitk::Image* segmentationImage = dynamic_cast<mitk::Image*>(newSegmentation->GetData());
      AccessFixedDimensionByItk_1(segmentationImage, ITKImageIsEmpty, 3, imageIsEmpty);

      if (!imageIsEmpty)
      {
        QMessageBox::StandardButton answer = QMessageBox::question(this->GetGUI()->GetParent(),
            "Start morphological segmentation",
            "You are about to start the morphological segmentation pipeline "
            "on an existing, non-empty image. The current mask needs to be erased.\n"
            "\n"
            "Do you want to start the segmentation and wipe the current image?",
            QMessageBox::Yes|QMessageBox::No);

        if (answer == QMessageBox::No)
        {
          return;
        }
      }
    }
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

  mitk::DataNode::Pointer axialCutOffPlaneNode = this->CreateAxialCutOffPlaneNode(referenceImage);
  this->GetDataStorage()->Add(axialCutOffPlaneNode, newSegmentation);

  this->WaitCursorOn();

  // Mark the newSegmentation as "unfinished".
  newSegmentation->SetBoolProperty(MorphologicalSegmentorPipelineManager::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED.c_str(), false);

  try
  {
    // Create that orange colour that MIDAS uses to highlight edited regions.
    mitk::ColorProperty::Pointer col = mitk::ColorProperty::New();
    col->SetColor(1.0f, static_cast<float>(165.0 / 255.0), 0.0f);

    // Create additions data node, and store reference to image
    float segCol[3];
    newSegmentation->GetColor(segCol);
    mitk::ColorProperty::Pointer segmentationColor = mitk::ColorProperty::New(segCol[0], segCol[1], segCol[2]);


    // Create extra data and store with ToolManager
    ITKRegionParametersDataNodeProperty::Pointer erodeAddEditingProp = ITKRegionParametersDataNodeProperty::New();
    erodeAddEditingProp->SetSize(1,1,1);
    erodeAddEditingProp->SetValid(false);
    mitk::DataNode::Pointer erodeAddNode = paintbrushTool->CreateEmptySegmentationNode(referenceImage, PaintbrushTool::EROSIONS_ADDITIONS_NAME, col->GetColor());
    erodeAddNode->SetBoolProperty("helper object", true);
    erodeAddNode->SetBoolProperty("visible", false);
    erodeAddNode->SetColor(segCol);
    erodeAddNode->SetProperty("binaryimage.selectedcolor", segmentationColor);
    erodeAddNode->AddProperty(PaintbrushTool::REGION_PROPERTY_NAME.c_str(), erodeAddEditingProp);

    ITKRegionParametersDataNodeProperty::Pointer erodeSubtractEditingProp = ITKRegionParametersDataNodeProperty::New();
    erodeSubtractEditingProp->SetSize(1,1,1);
    erodeSubtractEditingProp->SetValid(false);
    mitk::DataNode::Pointer erodeSubtractNode = paintbrushTool->CreateEmptySegmentationNode( referenceImage, PaintbrushTool::EROSIONS_SUBTRACTIONS_NAME, col->GetColor());
    erodeSubtractNode->SetBoolProperty("helper object", true);
    erodeSubtractNode->SetBoolProperty("visible", false);
    erodeSubtractNode->SetColor(col->GetColor());
    erodeSubtractNode->SetProperty("binaryimage.selectedcolor", col);
    erodeSubtractNode->AddProperty(PaintbrushTool::REGION_PROPERTY_NAME.c_str(), erodeSubtractEditingProp);

    ITKRegionParametersDataNodeProperty::Pointer dilateAddEditingProp = ITKRegionParametersDataNodeProperty::New();
    dilateAddEditingProp->SetSize(1,1,1);
    dilateAddEditingProp->SetValid(false);
    mitk::DataNode::Pointer dilateAddNode = paintbrushTool->CreateEmptySegmentationNode( referenceImage, PaintbrushTool::DILATIONS_ADDITIONS_NAME, col->GetColor());
    dilateAddNode->SetBoolProperty("helper object", true);
    dilateAddNode->SetBoolProperty("visible", false);
    dilateAddNode->SetColor(segCol);
    dilateAddNode->SetProperty("binaryimage.selectedcolor", segmentationColor);
    dilateAddNode->AddProperty(PaintbrushTool::REGION_PROPERTY_NAME.c_str(), dilateAddEditingProp);

    ITKRegionParametersDataNodeProperty::Pointer dilateSubtractEditingProp = ITKRegionParametersDataNodeProperty::New();
    dilateSubtractEditingProp->SetSize(1,1,1);
    dilateSubtractEditingProp->SetValid(false);
    mitk::DataNode::Pointer dilateSubtractNode = paintbrushTool->CreateEmptySegmentationNode( referenceImage, PaintbrushTool::DILATIONS_SUBTRACTIONS_NAME, col->GetColor());
    dilateSubtractNode->SetBoolProperty("helper object", true);
    dilateSubtractNode->SetBoolProperty("visible", false);
    dilateSubtractNode->SetColor(col->GetColor());
    dilateSubtractNode->SetProperty("binaryimage.selectedcolor", col);
    dilateSubtractNode->AddProperty(PaintbrushTool::REGION_PROPERTY_NAME.c_str(), dilateSubtractEditingProp);

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
    // 0. The Zeroth image is the segmentation image itself. Although it is not used by the paintbrush tool,
    //    this is assumed by the image selector widget.
    // 1. The First image is the "Additions" image for erosions, that we can manually add data/voxels to.
    // 2. The Second image is the "Subtractions" image for erosions, that is used for connection breaker.
    // 3. The Third image is the "Additions" image for dilations, that we can manually add data/voxels to.
    // 4. The Forth image is the "Subtractions" image for dilations, that is used for connection breaker.
    //
    // This must match the order in:
    //
    // 1. niftkMorphologicalSegmentorPipelineManager::UpdateSegmentation()
    // 2. niftk::PaintbrushTool.
    // and unit tests etc. Probably best to search for
    // MORPH_EDITS_EROSIONS_SUBTRACTIONS
    // MORPH_EDITS_EROSIONS_ADDITIONS
    // MORPH_EDITS_DILATIONS_SUBTRACTIONS
    // MORPH_EDITS_DILATIONS_ADDITIONS

    std::vector<mitk::DataNode*> workingNodes(5);
    workingNodes[PaintbrushTool::SEGMENTATION] = newSegmentation;
    workingNodes[PaintbrushTool::EROSIONS_ADDITIONS] = erodeAddNode;
    workingNodes[PaintbrushTool::EROSIONS_SUBTRACTIONS] = erodeSubtractNode;
    workingNodes[PaintbrushTool::DILATIONS_ADDITIONS] = dilateAddNode;
    workingNodes[PaintbrushTool::DILATIONS_SUBTRACTIONS] = dilateSubtractNode;

    toolManager->SetWorkingData(workingNodes);

    /// Note:
    /// The tool selection box tracks the events when the working data changes,
    /// and enables and disables the tool buttons accordingly. However, we want
    /// to enable the buttons only in the second and third step of the workflow
    /// (erosion and dilation). So, since we are at the first step, we have to
    /// disable the tool selector that was enabled at the previous call when
    /// setting the working data.
    m_MorphologicalSegmentorGUI->SetToolSelectorEnabled(false);

    // Set properties, and then the control values to match.
    if (isRestarting)
    {
      if (newSegmentation->GetProperty("midas.morph.stage"))
      {
        /// The morphological segmentor pipeline has already be performed on this data, and the
        /// pipeline parameters are stored in the data node. We need to relaunch the pipeline
        /// up to the step where it was finished last time.
        newSegmentation->SetBoolProperty("midas.morph.restarting", true);
        this->SetControlsFromSegmentationNodeProps();
      }
      else
      {
        /// The segmentation is started on an already existing binary image, but the pipeline
        /// has not been performed on this data. The image contents need to be erased and we
        /// need to set the initial parameters based on the reference image.
        mitk::Image* segmentationImage = dynamic_cast<mitk::Image*>(newSegmentation->GetData());
        AccessFixedDimensionByItk(segmentationImage, ITKClearImage, 3);
        this->SetControlsFromReferenceImage();
      }
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

  if (!isRestarting)
  {
    this->GetView()->SetDataManagerSelection(newSegmentation);
  } 
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
      PaintbrushTool::Pointer paintbrushTool = this->GetToolByType<PaintbrushTool>();

      mitk::DataNode::Pointer erodeSubtractNode = this->GetToolManager()->GetWorkingData(PaintbrushTool::EROSIONS_SUBTRACTIONS);
      mitk::DataNode::Pointer dilateSubtractNode = this->GetToolManager()->GetWorkingData(PaintbrushTool::DILATIONS_SUBTRACTIONS);

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
      this->OnActiveToolChanged(); // make sure we de-activate tools.
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
    this->OnActiveToolChanged();
    m_MorphologicalSegmentorGUI->EnableSegmentationWidgets(false);
    m_MorphologicalSegmentorGUI->SetTabIndex(0);
    m_PipelineManager->FinalizeSegmentation();

    /// Remove the axial cut-off plane node from the data storage.
    mitk::DataNode::Pointer axialCutOffPlaneNode = this->GetDataStorage()->GetNamedDerivedNode("Axial cut-off plane", segmentationNode);
    this->GetDataStorage()->Remove(axialCutOffPlaneNode);

    this->GetView()->SetDataManagerSelection(this->GetReferenceNode());
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
    this->OnActiveToolChanged();
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

    this->GetView()->SetDataManagerSelection(segmentationNode);
    this->RequestRenderWindowUpdate();
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnCancelButtonClicked()
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNode();
  if (segmentationNode.IsNotNull())
  {
    this->OnActiveToolChanged();
    m_MorphologicalSegmentorGUI->EnableSegmentationWidgets(false);
    m_MorphologicalSegmentorGUI->SetTabIndex(0);
    m_PipelineManager->RemoveWorkingNodes();
    mitk::Image::Pointer segmentationImage = dynamic_cast<mitk::Image*>(segmentationNode->GetData());
    m_PipelineManager->DestroyPipeline(segmentationImage);
    mitk::DataNode::Pointer axialCutOffPlaneNode = this->GetDataStorage()->GetNamedDerivedNode("Axial cut-off plane", segmentationNode);
    this->GetDataStorage()->Remove(axialCutOffPlaneNode);
    this->GetDataStorage()->Remove(segmentationNode);
    this->GetView()->SetDataManagerSelection(this->GetReferenceNode());
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
    mitk::DataNode* node = this->GetWorkingNode(imageIndex);
    assert(node);
    ITKRegionParametersDataNodeProperty::Pointer prop =
        dynamic_cast<ITKRegionParametersDataNodeProperty*>(node->GetProperty(PaintbrushTool::REGION_PROPERTY_NAME.c_str()));
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
    this->OnActiveToolChanged();
    m_MorphologicalSegmentorGUI->EnableSegmentationWidgets(false);
    m_MorphologicalSegmentorGUI->SetTabIndex(0);

    mitk::DataNode::Pointer axialCutOffPlaneNode = this->GetDataStorage()->GetNamedDerivedNode("Axial cut-off plane", segmentationNode);
    if (axialCutOffPlaneNode.IsNotNull())
    {
      this->GetDataStorage()->Remove(axialCutOffPlaneNode);
    }
    m_PipelineManager->RemoveWorkingNodes();
    mitk::Image::Pointer segmentationImage = dynamic_cast<mitk::Image*>(segmentationNode->GetData());
    m_PipelineManager->DestroyPipeline(segmentationImage);
    this->GetView()->SetDataManagerSelection(this->GetReferenceNode());
    this->RequestRenderWindowUpdate();
    mitk::UndoController::GetCurrentUndoModel()->Clear();
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnNodeVisibilityChanged(const mitk::DataNode* node, const mitk::BaseRenderer* /*renderer*/)
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNode();

  std::vector<mitk::DataNode*> workingNodes = this->GetWorkingNodes();
  if (segmentationNode.IsNotNull() && node == segmentationNode && workingNodes.size() == 5)
  {
    mitk::DataNode::Pointer axialCutOffPlaneNode = this->GetDataStorage()->GetNamedDerivedNode("Axial cut-off plane", segmentationNode);

    bool segmentationNodeVisibility;
    if (node->GetVisibility(segmentationNodeVisibility, 0) && segmentationNodeVisibility)
    {
      int tabIndex = m_MorphologicalSegmentorGUI->GetTabIndex();
      workingNodes[PaintbrushTool::EROSIONS_ADDITIONS]->SetVisibility(false);
      workingNodes[PaintbrushTool::EROSIONS_SUBTRACTIONS]->SetVisibility(tabIndex == 1);
      workingNodes[PaintbrushTool::DILATIONS_ADDITIONS]->SetVisibility(false);
      workingNodes[PaintbrushTool::DILATIONS_SUBTRACTIONS]->SetVisibility(tabIndex == 2);
      axialCutOffPlaneNode->SetVisibility(true);
    }
    else
    {
      for (std::size_t i = 1; i < workingNodes.size(); ++i)
      {
        workingNodes[i]->SetVisibility(false);
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
