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

  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(ThresholdingValuesChanged(double, double)), SLOT(OnThresholdingValuesChanged(double, double)));
  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(AxialCutOffSliceNumberChanged(int)), SLOT(OnAxialCutOffSliceNumberChanged(int)));
  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(ErosionsValuesChanged(double, int)), SLOT(OnErosionsValuesChanged(double, int)));
  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(DilationsValuesChanged(double, double, int)), SLOT(OnDilationsValuesChanged(double, double, int)));
  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(RethresholdingValuesChanged(int)), SLOT(OnRethresholdingValuesChanged(int)));
  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(TabChanged(int)), SLOT(OnTabChanged(int)));
  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(OKButtonClicked()), SLOT(OnOKButtonClicked()));
  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(CancelButtonClicked()), SLOT(OnCancelButtonClicked()));
  this->connect(m_MorphologicalSegmentorGUI, SIGNAL(RestartButtonClicked()), SLOT(OnRestartButtonClicked()));
}


//-----------------------------------------------------------------------------
bool MorphologicalSegmentorController::IsASegmentationImage(const mitk::DataNode* node)
{
  assert(node);

  /// It needs to hold a binary image.
  if (!niftk::IsNodeABinaryImage(node))
  {
    return false;
  }

  mitk::DataStorage* dataStorage = this->GetDataStorage();

  /// Its parent node needs to hold a grey scale image.
  if (!niftk::FindFirstParentImage(dataStorage, node, false))
  {
    return false;
  }

  /// It should also have four children for the paintbrush tool and one for the
  /// axial cut-off plane node.

  std::set<std::string> set;

  mitk::DataStorage::SetOfObjects::ConstPointer children = dataStorage->GetDerivations(node);
  for (auto it = children->Begin(); it != children->End(); ++it)
  {
    set.insert(it->Value()->GetName());
  }

  return set.find(PaintbrushTool::EROSIONS_SUBTRACTIONS_NAME) != set.end()
      && set.find(PaintbrushTool::EROSIONS_ADDITIONS_NAME) != set.end()
      && set.find(PaintbrushTool::DILATIONS_SUBTRACTIONS_NAME) != set.end()
      && set.find(PaintbrushTool::DILATIONS_ADDITIONS_NAME) != set.end()
      && set.find(PaintbrushTool::AXIAL_CUT_OFF_PLANE_NAME) != set.end();
}


//-----------------------------------------------------------------------------
bool MorphologicalSegmentorController::IsAWorkingImage(const mitk::DataNode* node)
{
  assert(node);
  bool result = false;

  if (niftk::IsNodeABinaryImage(node))
  {
    mitk::DataNode::Pointer parent = niftk::FindFirstParentImage(this->GetDataStorage(), node, true);
    if (parent.IsNotNull())
    {
      std::string name;
      if (node->GetStringProperty("name", name))
      {
        if (   name == PaintbrushTool::EROSIONS_SUBTRACTIONS_NAME
            || name == PaintbrushTool::EROSIONS_ADDITIONS_NAME
            || name == PaintbrushTool::DILATIONS_SUBTRACTIONS_NAME
            || name == PaintbrushTool::DILATIONS_ADDITIONS_NAME
            )
        {
          result = true;
        }
      }
    }
  }

  return result;
}


//-----------------------------------------------------------------------------
std::vector<mitk::DataNode*> MorphologicalSegmentorController::GetWorkingNodesFromSegmentationNode(mitk::DataNode* segmentationNode)
{
  assert(segmentationNode);

  std::vector<mitk::DataNode*> workingNodes(6);
  std::fill(workingNodes.begin(), workingNodes.end(), nullptr);

  workingNodes[PaintbrushTool::SEGMENTATION] = segmentationNode;

  mitk::DataStorage::SetOfObjects::ConstPointer derivedNodes = this->GetDataStorage()->GetDerivations(segmentationNode);

  for (auto it = derivedNodes->Begin(); it != derivedNodes->End(); ++it)
  {
    mitk::DataNode* derivedNode = it->Value();
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
    else if (name == PaintbrushTool::AXIAL_CUT_OFF_PLANE_NAME)
    {
      workingNodes[PaintbrushTool::AXIAL_CUT_OFF_PLANE] = derivedNode;
    }
  }

  if (std::count(workingNodes.begin(), workingNodes.end(), nullptr) != 0)
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

  this->WaitCursorOn();

  // Mark the newSegmentation as "unfinished".
  newSegmentation->SetBoolProperty("midas.morph.finished", false);

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

    mitk::DataNode::Pointer axialCutOffPlaneNode = this->CreateAxialCutOffPlaneNode(referenceImage);

    // Add the image to data storage, and specify this derived image as the one the toolManager will edit to.
    this->GetDataStorage()->Add(erodeAddNode, newSegmentation); // add as a child, because the segmentation "derives" from the original
    this->GetDataStorage()->Add(erodeSubtractNode, newSegmentation); // add as a child, because the segmentation "derives" from the original
    this->GetDataStorage()->Add(dilateAddNode, newSegmentation); // add as a child, because the segmentation "derives" from the original
    this->GetDataStorage()->Add(dilateSubtractNode, newSegmentation); // add as a child, because the segmentation "derives" from the original
    this->GetDataStorage()->Add(axialCutOffPlaneNode, newSegmentation);

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

    std::vector<mitk::DataNode*> workingNodes(6);
    workingNodes[PaintbrushTool::SEGMENTATION] = newSegmentation;
    workingNodes[PaintbrushTool::EROSIONS_ADDITIONS] = erodeAddNode;
    workingNodes[PaintbrushTool::EROSIONS_SUBTRACTIONS] = erodeSubtractNode;
    workingNodes[PaintbrushTool::DILATIONS_ADDITIONS] = dilateAddNode;
    workingNodes[PaintbrushTool::DILATIONS_SUBTRACTIONS] = dilateSubtractNode;
    workingNodes[PaintbrushTool::AXIAL_CUT_OFF_PLANE] = axialCutOffPlaneNode;

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
        this->SetControlsFromSegmentationNode();
        int axialCutOffSlice;
        if (newSegmentation->GetIntProperty("midas.morph.thresholding.slice", axialCutOffSlice))
        {
          this->UpdateAxialCutOffPlaneNode(axialCutOffSlice);
        }
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
      this->SetControlsFromSegmentationNode();
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
void MorphologicalSegmentorController::OnReferenceNodesChanged()
{
  this->SetControlsFromReferenceImage();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnWorkingNodesChanged()
{
  this->SetControlsFromSegmentationNode();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnThresholdingValuesChanged(double lowerThreshold, double upperThreshold)
{
  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  if (segmentationNode)
  {
    segmentationNode->SetFloatProperty("midas.morph.thresholding.lower", lowerThreshold);
    segmentationNode->SetFloatProperty("midas.morph.thresholding.upper", upperThreshold);
    m_PipelineManager->UpdateSegmentation();

    this->RequestRenderWindowUpdate();
  }
}


void MorphologicalSegmentorController::OnAxialCutOffSliceNumberChanged(int axialSliceNumber)
{
  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  if (segmentationNode)
  {
    segmentationNode->SetIntProperty("midas.morph.thresholding.slice", axialSliceNumber);
    m_PipelineManager->UpdateSegmentation();
  }

  this->UpdateAxialCutOffPlaneNode(axialSliceNumber);
}


void MorphologicalSegmentorController::UpdateAxialCutOffPlaneNode(int axialSliceNumber)
{
  const mitk::Image* referenceImage = this->GetReferenceImage();
  mitk::BaseGeometry* geometry = referenceImage->GetGeometry();

  int axialAxis = niftk::GetThroughPlaneAxis(referenceImage, IMAGE_ORIENTATION_AXIAL);
  int axialUpDirection = niftk::GetUpDirection(referenceImage, IMAGE_ORIENTATION_AXIAL);

  mitk::DataNode* axialCutOffPlaneNode = this->GetWorkingNode(PaintbrushTool::AXIAL_CUT_OFF_PLANE);
  mitk::Plane* axialCutOffPlane = dynamic_cast<mitk::Plane*>(axialCutOffPlaneNode->GetData());

  // Lift the axial cut-off plane to the height determined by axialSliceNumber.
  mitk::Point3D planeCentre = axialCutOffPlane->GetGeometry()->GetOrigin();
  planeCentre[2] = geometry->GetOrigin()[2] + (axialUpDirection * axialSliceNumber - 0.5) * geometry->GetSpacing()[axialAxis];
  axialCutOffPlane->SetOrigin(planeCentre);

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnErosionsValuesChanged(double upperThreshold, int numberOfErosions)
{
  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  if (segmentationNode)
  {
    segmentationNode->SetFloatProperty("midas.morph.erosion.threshold", upperThreshold);
    segmentationNode->SetIntProperty("midas.morph.erosion.iterations", numberOfErosions);
    m_PipelineManager->UpdateSegmentation();
  }
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnDilationsValuesChanged(double lowerPercentage, double upperPercentage, int numberOfDilations)
{
  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  if (segmentationNode)
  {
    segmentationNode->SetFloatProperty("midas.morph.dilation.lower", lowerPercentage);
    segmentationNode->SetFloatProperty("midas.morph.dilation.upper", upperPercentage);
    segmentationNode->SetIntProperty("midas.morph.dilation.iterations", numberOfDilations);
    m_PipelineManager->UpdateSegmentation();
  }
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnRethresholdingValuesChanged(int boxSize)
{
  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  if (segmentationNode)
  {
    segmentationNode->SetIntProperty("midas.morph.rethresholding.box", boxSize);
    m_PipelineManager->UpdateSegmentation();
  }
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnTabChanged(int tabIndex)
{
  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  if (segmentationNode)
  {
    if (tabIndex == 1 || tabIndex == 2)
    {
      m_MorphologicalSegmentorGUI->SetToolSelectorEnabled(true);

      PaintbrushTool::Pointer paintbrushTool = this->GetToolByType<PaintbrushTool>();

      mitk::DataNode* erodeSubtractNode = this->GetWorkingNode(PaintbrushTool::EROSIONS_SUBTRACTIONS);
      mitk::DataNode* dilateSubtractNode = this->GetWorkingNode(PaintbrushTool::DILATIONS_SUBTRACTIONS);

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

    segmentationNode->SetIntProperty("midas.morph.stage", tabIndex);
    m_PipelineManager->UpdateSegmentation();

    this->RequestRenderWindowUpdate();
  }

  m_TabIndex = tabIndex;
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnOKButtonClicked()
{
  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  if (segmentationNode)
  {
    this->OnActiveToolChanged();
    m_PipelineManager->FinalizeSegmentation();

    segmentationNode->SetBoolProperty("midas.morph.finished", true);
    segmentationNode->SetIntProperty("midas.morph.stage", 0);

    this->RemoveWorkingNodes();
    m_MorphologicalSegmentorGUI->SetControlsFromSegmentationNode(nullptr);

    this->GetView()->SetDataManagerSelection(this->GetReferenceNode());
    this->RequestRenderWindowUpdate();
    mitk::UndoController::GetCurrentUndoModel()->Clear();
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnRestartButtonClicked()
{
  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  if (segmentationNode)
  {
    this->OnActiveToolChanged();
    m_PipelineManager->ClearWorkingData();
    this->SetSegmentationNodePropsFromReferenceImage();
    this->SetControlsFromReferenceImage();
    this->SetControlsFromSegmentationNode();
    m_PipelineManager->UpdateSegmentation();

//    /// Reset the axial cut-off plane to the bottom of the image.
    {
      mitk::DataNode* referenceNode = this->GetReferenceNode();
      mitk::Image* referenceImage = dynamic_cast<mitk::Image*>(referenceNode->GetData());
      mitk::BaseGeometry* geometry = referenceImage->GetGeometry();

      mitk::DataNode* axialCutOffPlaneNode = this->GetWorkingNode(PaintbrushTool::AXIAL_CUT_OFF_PLANE);
      mitk::Plane* axialCutOffPlane = dynamic_cast<mitk::Plane*>(axialCutOffPlaneNode->GetData());

      int axialAxis = GetThroughPlaneAxis(referenceImage, IMAGE_ORIENTATION_AXIAL);
      int axialUpDirection = niftk::GetUpDirection(referenceImage, IMAGE_ORIENTATION_AXIAL);

      // The centre of the plane is the same as the centre of the image, but it is shifted
      // along the axial axis to a position determined by axialSliceNumber.
      int axialSliceNumber = 0;
      segmentationNode->GetIntProperty("midas.morph.thresholding.slice", axialSliceNumber);

      mitk::Point3D planeCentre = geometry->GetOrigin();
      planeCentre[2] = geometry->GetOrigin()[2] + (axialUpDirection * axialSliceNumber - 0.5) * geometry->GetSpacing()[axialAxis];

      axialCutOffPlane->SetOrigin(planeCentre);
   }

    this->GetView()->SetDataManagerSelection(segmentationNode);
    this->RequestRenderWindowUpdate();
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnCancelButtonClicked()
{
  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  if (segmentationNode)
  {
    this->OnActiveToolChanged();
    m_MorphologicalSegmentorGUI->EnableSegmentationWidgets(false);
    m_MorphologicalSegmentorGUI->SetTabIndex(0);
    this->RemoveWorkingNodes();
    mitk::Image* segmentationImage = dynamic_cast<mitk::Image*>(segmentationNode->GetData());
    m_PipelineManager->DestroyPipeline(segmentationImage);
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
  if  (this->GetWorkingNode())
  {
    this->OnCancelButtonClicked();
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnSegmentationEdited(int imageIndex)
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


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnNodeRemoved(const mitk::DataNode* removedNode)
{
  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  if (segmentationNode && segmentationNode == removedNode)
  {
    this->OnActiveToolChanged();
    m_MorphologicalSegmentorGUI->EnableSegmentationWidgets(false);
    m_MorphologicalSegmentorGUI->SetTabIndex(0);

    this->RemoveWorkingNodes();
    mitk::Image* segmentationImage = dynamic_cast<mitk::Image*>(segmentationNode->GetData());
    m_PipelineManager->DestroyPipeline(segmentationImage);
    this->GetView()->SetDataManagerSelection(this->GetReferenceNode());
    this->RequestRenderWindowUpdate();
    mitk::UndoController::GetCurrentUndoModel()->Clear();
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::RemoveWorkingNodes()
{
  std::vector<mitk::DataNode*> workingNodes = this->GetWorkingNodes();

  mitk::ToolManager* toolManager = this->GetToolManager();

  std::vector<mitk::DataNode*> noWorkingNodes(0);
  toolManager->SetWorkingData(noWorkingNodes);

  for (unsigned i = 1; i < workingNodes.size(); ++i)
  {
    this->GetDataStorage()->Remove(workingNodes[i]);
  }

  toolManager->ActivateTool(-1);
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::OnNodeVisibilityChanged(const mitk::DataNode* node, const mitk::BaseRenderer* /*renderer*/)
{
  mitk::DataNode* segmentationNode = this->GetWorkingNode();

  std::vector<mitk::DataNode*> workingNodes = this->GetWorkingNodes();
  if (segmentationNode && node == segmentationNode && workingNodes.size() == 6)
  {
    bool segmentationNodeVisibility;
    if (node->GetVisibility(segmentationNodeVisibility, 0) && segmentationNodeVisibility)
    {
      int tabIndex = m_MorphologicalSegmentorGUI->GetTabIndex();
      workingNodes[PaintbrushTool::EROSIONS_ADDITIONS]->SetVisibility(false);
      workingNodes[PaintbrushTool::EROSIONS_SUBTRACTIONS]->SetVisibility(tabIndex == 1);
      workingNodes[PaintbrushTool::DILATIONS_ADDITIONS]->SetVisibility(false);
      workingNodes[PaintbrushTool::DILATIONS_SUBTRACTIONS]->SetVisibility(tabIndex == 2);
      workingNodes[PaintbrushTool::AXIAL_CUT_OFF_PLANE]->SetVisibility(true);
    }
    else
    {
      for (std::size_t i = 1; i < workingNodes.size(); ++i)
      {
        workingNodes[i]->SetVisibility(false);
      }
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
  axialCutOffPlaneNode->SetName(PaintbrushTool::AXIAL_CUT_OFF_PLANE_NAME);
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
  const mitk::Image* referenceImage = this->GetReferenceImage();
  mitk::DataNode* segmentationNode = this->GetWorkingNode();

  if (referenceImage && segmentationNode)
  {
    int thresholdingSlice = 0;
    int upDirection = GetUpDirection(referenceImage, IMAGE_ORIENTATION_AXIAL);
    if (upDirection == -1)
    {
      int axialAxis = GetThroughPlaneAxis(referenceImage, IMAGE_ORIENTATION_AXIAL);
      thresholdingSlice = referenceImage->GetDimension(axialAxis) - 1;
    }

    segmentationNode->SetIntProperty("midas.morph.stage", 0);
    segmentationNode->SetFloatProperty("midas.morph.thresholding.lower", referenceImage->GetStatistics()->GetScalarValueMin());
    segmentationNode->SetFloatProperty("midas.morph.thresholding.upper", referenceImage->GetStatistics()->GetScalarValueMin());
    segmentationNode->SetIntProperty("midas.morph.thresholding.slice", thresholdingSlice);
    segmentationNode->SetFloatProperty("midas.morph.erosion.threshold", referenceImage->GetStatistics()->GetScalarValueMax());
    segmentationNode->SetIntProperty("midas.morph.erosion.iterations", 0);
    segmentationNode->SetFloatProperty("midas.morph.dilation.lower", 60);
    segmentationNode->SetFloatProperty("midas.morph.dilation.upper", 160);
    segmentationNode->SetIntProperty("midas.morph.dilation.iterations", 0);
    segmentationNode->SetIntProperty("midas.morph.rethresholding.box", 0);
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::SetControlsFromReferenceImage()
{
  double lowestValue = 0.0;
  double highestValue = 1.0;
  int axialAxis = 0;
  int numberOfAxialSlices = 1;
  int upDirection = 1;

  if (auto referenceImage = this->GetReferenceImage())
  {
    auto statistics = referenceImage->GetStatistics();
    lowestValue = statistics->GetScalarValueMin();
    highestValue = statistics->GetScalarValueMax();
    axialAxis = this->GetReferenceImageSliceAxis(IMAGE_ORIENTATION_AXIAL);
    numberOfAxialSlices = referenceImage->GetDimension(axialAxis);
    upDirection = niftk::GetUpDirection(referenceImage, IMAGE_ORIENTATION_AXIAL);
  }

  m_MorphologicalSegmentorGUI->SetControlsByReferenceImage(
      lowestValue,
      highestValue,
      numberOfAxialSlices,
      upDirection);
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorController::SetControlsFromSegmentationNode()
{
  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  m_MorphologicalSegmentorGUI->SetControlsFromSegmentationNode(segmentationNode);
}

}
