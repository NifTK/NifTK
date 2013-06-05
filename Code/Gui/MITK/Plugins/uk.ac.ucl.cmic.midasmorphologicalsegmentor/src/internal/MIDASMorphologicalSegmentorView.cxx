/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "MIDASMorphologicalSegmentorView.h"

#include <QMessageBox>

#include <berryIWorkbenchPage.h>
#include <mitkImageAccessByItk.h>
#include <mitkITKImageImport.h>
#include <mitkImage.h>
#include <mitkImageCast.h>
#include <mitkImageStatisticsHolder.h>
#include <mitkColorProperty.h>
#include <mitkDataStorageUtils.h>
#include <mitkUndoController.h>
#include <mitkMIDASOrientationUtils.h>

#include <itkConversionUtils.h>
#include <mitkITKRegionParametersDataNodeProperty.h>
#include <mitkMIDASTool.h>
#include <mitkMIDASPaintbrushTool.h>

const std::string MIDASMorphologicalSegmentorView::VIEW_ID = "uk.ac.ucl.cmic.midasmorphologicalsegmentor";

//-----------------------------------------------------------------------------
MIDASMorphologicalSegmentorView::MIDASMorphologicalSegmentorView()
: QmitkMIDASBaseSegmentationFunctionality()
, m_Layout(NULL)
, m_ContainerForControlsWidget(NULL)
, m_MorphologicalControls(NULL)
, m_PipelineManager(NULL)
, m_TabCounter(-1)
{
}


//-----------------------------------------------------------------------------
MIDASMorphologicalSegmentorView::MIDASMorphologicalSegmentorView(
    const MIDASMorphologicalSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
MIDASMorphologicalSegmentorView::~MIDASMorphologicalSegmentorView()
{
}


//-----------------------------------------------------------------------------
std::string MIDASMorphologicalSegmentorView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::ClosePart()
{
  if (m_PipelineManager->HasSegmentationNode())
  {
    this->OnCancelButtonClicked();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnCreateNewSegmentationButtonPressed()
{
  // Create the new segmentation, either using a previously selected one, or create a new volume.
  mitk::DataNode::Pointer newSegmentation = NULL;
  bool isRestarting = false;

  // Make sure we have a reference images... which should always be true at this point.
  mitk::Image::Pointer image = m_PipelineManager->GetReferenceImageFromToolManager(0);
  if (image.IsNotNull())
  {

    // Make sure we can retrieve the paintbrush tool, which can be used to create a new segmentation image.
    mitk::ToolManager* toolManager = this->GetToolManager();
    assert(toolManager);

    int paintbrushToolId = toolManager->GetToolIdByToolType<mitk::MIDASPaintbrushTool>();

    mitk::Tool* paintbrushTool = toolManager->GetToolById(paintbrushToolId);
    assert(paintbrushTool);

    if (mitk::IsNodeABinaryImage(m_SelectedNode)
        && this->CanStartSegmentationForBinaryNode(m_SelectedNode)
        && !this->IsNodeASegmentationImage(m_SelectedNode)
        )
    {
      newSegmentation =  m_SelectedNode;
      isRestarting = true;
    }
    else
    {
      newSegmentation = CreateNewSegmentation(m_DefaultSegmentationColor);

      // The above method returns NULL if the user exited the colour selection dialog box.
      if (newSegmentation.IsNull())
      {
        return;
      }
    }

    this->WaitCursorOn();

    // Mark the newSegmentation as "unfinished".
    newSegmentation->SetProperty(mitk::MIDASMorphologicalSegmentorPipelineManager::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED.c_str(), mitk::BoolProperty::New(false));

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
      mitk::DataNode::Pointer erodeAddNode = paintbrushTool->CreateEmptySegmentationNode( image, mitk::MIDASTool::MORPH_EDITS_EROSIONS_ADDITIONS, col->GetColor());
      erodeAddNode->SetBoolProperty("helper object", true);
      erodeAddNode->SetBoolProperty("visible", false);
      erodeAddNode->SetColor(segCol);
      erodeAddNode->SetProperty("binaryimage.selectedcolor", segmentationColor);
      erodeAddNode->AddProperty(mitk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str(), erodeAddEditingProp);

      mitk::ITKRegionParametersDataNodeProperty::Pointer erodeSubtractEditingProp = mitk::ITKRegionParametersDataNodeProperty::New();
      erodeSubtractEditingProp->SetSize(1,1,1);
      erodeSubtractEditingProp->SetValid(false);
      mitk::DataNode::Pointer erodeSubtractNode = paintbrushTool->CreateEmptySegmentationNode( image, mitk::MIDASTool::MORPH_EDITS_EROSIONS_SUBTRACTIONS, col->GetColor());
      erodeSubtractNode->SetBoolProperty("helper object", true);
      erodeSubtractNode->SetBoolProperty("visible", false);
      erodeSubtractNode->SetColor(col->GetColor());
      erodeSubtractNode->SetProperty("binaryimage.selectedcolor", col);
      erodeSubtractNode->AddProperty(mitk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str(), erodeSubtractEditingProp);

      mitk::ITKRegionParametersDataNodeProperty::Pointer dilateAddEditingProp = mitk::ITKRegionParametersDataNodeProperty::New();
      dilateAddEditingProp->SetSize(1,1,1);
      dilateAddEditingProp->SetValid(false);
      mitk::DataNode::Pointer dilateAddNode = paintbrushTool->CreateEmptySegmentationNode( image, mitk::MIDASTool::MORPH_EDITS_DILATIONS_ADDITIONS, col->GetColor());
      dilateAddNode->SetBoolProperty("helper object", true);
      dilateAddNode->SetBoolProperty("visible", false);
      dilateAddNode->SetColor(segCol);
      dilateAddNode->SetProperty("binaryimage.selectedcolor", segmentationColor);
      dilateAddNode->AddProperty(mitk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str(), dilateAddEditingProp);

      mitk::ITKRegionParametersDataNodeProperty::Pointer dilateSubtractEditingProp = mitk::ITKRegionParametersDataNodeProperty::New();
      dilateSubtractEditingProp->SetSize(1,1,1);
      dilateSubtractEditingProp->SetValid(false);
      mitk::DataNode::Pointer dilateSubtractNode = paintbrushTool->CreateEmptySegmentationNode( image, mitk::MIDASTool::MORPH_EDITS_DILATIONS_SUBTRACTIONS, col->GetColor());
      dilateSubtractNode->SetBoolProperty("helper object", true);
      dilateSubtractNode->SetBoolProperty("visible", false);
      dilateSubtractNode->SetColor(col->GetColor());
      dilateSubtractNode->SetProperty("binaryimage.selectedcolor", col);
      dilateSubtractNode->AddProperty(mitk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str(), dilateSubtractEditingProp);

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
      // 1. MIDASMorphologicalSegmentorPipelineManager::UpdateSegmentation()
      // 2. mitkMIDASPaintbrushTool.
      // and unit tests etc. Probably best to search for
      // MORPH_EDITS_EROSIONS_SUBTRACTIONS
      // MORPH_EDITS_EROSIONS_ADDITIONS
      // MORPH_EDITS_DILATIONS_SUBTRACTIONS
      // MORPH_EDITS_DILATIONS_ADDITIONS

      mitk::ToolManager::DataVectorType workingData;
      workingData.push_back(erodeAddNode);
      workingData.push_back(erodeSubtractNode);
      workingData.push_back(dilateAddNode);
      workingData.push_back(dilateSubtractNode);

      toolManager->SetWorkingData(workingData);

      // Set properties, and then the control values to match.
      if (isRestarting)
      {
        newSegmentation->SetBoolProperty("midas.morph.restarting", true);
      }
      else
      {
        this->SetDefaultParameterValuesFromReferenceImage();
        this->SetControlsByImageData();
      }
      this->SetControlsByParameterValues();
      m_PipelineManager->UpdateSegmentation();
    }
    catch (std::bad_alloc&)
    {
      QMessageBox::warning(NULL,"Create new segmentation","Could not allocate memory for new segmentation");
    }

    this->FocusOnCurrentWindow();
    this->RequestRenderWindowUpdate();
    this->WaitCursorOff();

  } // end if we have a reference image

  // Finally, select the new segmentation node.
  this->SetCurrentSelection(newSegmentation);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnThresholdingValuesChanged(double lowerThreshold, double upperThreshold, int axialSliceNumber)
{
  m_PipelineManager->OnThresholdingValuesChanged(lowerThreshold, upperThreshold, axialSliceNumber);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnErosionsValuesChanged(double upperThreshold, int numberOfErosions)
{
  m_PipelineManager->OnErosionsValuesChanged(upperThreshold, numberOfErosions);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnDilationValuesChanged(double lowerPercentage, double upperPercentage, int numberOfDilations)
{
  m_PipelineManager->OnDilationValuesChanged(lowerPercentage, upperPercentage, numberOfDilations);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnRethresholdingValuesChanged(int boxSize)
{
  m_PipelineManager->OnRethresholdingValuesChanged(boxSize);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnTabChanged(int i)
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNodeFromToolManager();
  if (segmentationNode.IsNotNull())
  {
    if (i == 1 || i == 2)
    {
      m_ToolSelector->SetEnabled(true);

      mitk::ToolManager::Pointer toolManager = this->GetToolManager();
      mitk::MIDASPaintbrushTool::Pointer paintbrushTool = dynamic_cast<mitk::MIDASPaintbrushTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASPaintbrushTool>()));

      mitk::DataNode::Pointer erodeSubtractNode = this->GetToolManager()->GetWorkingData(1);
      mitk::DataNode::Pointer dilateSubtractNode = this->GetToolManager()->GetWorkingData(3);

      if (i == 1)
      {
        paintbrushTool->SetErosionMode(true);
        erodeSubtractNode->SetVisibility(true);
        dilateSubtractNode->SetVisibility(false);

        // Only if we are switching from tab 2 to 1.
        if (m_TabCounter == 2)
        {
          mitk::Image* dilateSubtractImage = dynamic_cast<mitk::Image*>(dilateSubtractNode->GetData());
          mitk::Image* erodeSubtractImage = dynamic_cast<mitk::Image*>(erodeSubtractNode->GetData());
          if (dilateSubtractImage != NULL && erodeSubtractImage != NULL)
          {
            mitk::CopyIntensityData(dilateSubtractImage, erodeSubtractImage);
          }
        }
      }
      else // i==2
      {
        paintbrushTool->SetErosionMode(false);
        erodeSubtractNode->SetVisibility(false);
        dilateSubtractNode->SetVisibility(true);

        // Only if we are switching from tab 1 to 2.
        if (m_TabCounter == 1)
        {
          mitk::Image* erodeSubtractImage = dynamic_cast<mitk::Image*>(erodeSubtractNode->GetData());
          mitk::Image* dilateSubtractImage = dynamic_cast<mitk::Image*>(dilateSubtractNode->GetData());
          if (erodeSubtractImage != NULL && dilateSubtractImage != NULL)
          {
            mitk::CopyIntensityData(erodeSubtractImage, dilateSubtractImage);
          }
        }
      }
    }
    else
    {
      m_ToolSelector->SetEnabled(false);
      this->OnToolSelected(-1); // make sure we de-activate tools.
    }

    segmentationNode->SetIntProperty("midas.morph.stage", i);
    m_PipelineManager->UpdateSegmentation();
    this->RequestRenderWindowUpdate();
  }
  m_TabCounter = i;
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnOKButtonClicked()
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNodeFromToolManager();
  if (segmentationNode.IsNotNull())
  {
    this->OnToolSelected(-1);
    this->EnableSegmentationWidgets(false);
    m_MorphologicalControls->m_TabWidget->blockSignals(true);
    m_MorphologicalControls->m_TabWidget->setCurrentIndex(0);
    m_MorphologicalControls->m_TabWidget->blockSignals(false);
    m_PipelineManager->FinalizeSegmentation();
    this->FireNodeSelected(this->GetReferenceNodeFromToolManager());
    this->RequestRenderWindowUpdate();
    mitk::UndoController::GetCurrentUndoModel()->Clear();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnRestartButtonClicked()
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNodeFromToolManager();
  if (segmentationNode.IsNotNull())
  {
    this->OnToolSelected(-1);
    m_PipelineManager->ClearWorkingData();
    this->SetDefaultParameterValuesFromReferenceImage();
    this->SetControlsByImageData();
    this->SetControlsByParameterValues();
    m_PipelineManager->UpdateSegmentation();
    this->FireNodeSelected(segmentationNode);
    this->RequestRenderWindowUpdate();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnCancelButtonClicked()
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNodeFromToolManager();
  if (segmentationNode.IsNotNull())
  {
    this->OnToolSelected(-1);
    this->EnableSegmentationWidgets(false);
    m_MorphologicalControls->m_TabWidget->blockSignals(true);
    m_MorphologicalControls->m_TabWidget->setCurrentIndex(0);
    m_MorphologicalControls->m_TabWidget->blockSignals(false);
    m_PipelineManager->RemoveWorkingData();
    m_PipelineManager->DestroyPipeline();
    this->GetDataStorage()->Remove(segmentationNode);
    this->FireNodeSelected(this->GetReferenceNodeFromToolManager());
    this->RequestRenderWindowUpdate();
    mitk::UndoController::GetCurrentUndoModel()->Clear();
  }
}

//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::CreateQtPartControl(QWidget *parent)
{
  this->SetParent(parent);

  if (!m_MorphologicalControls)
  {
    m_Layout = new QGridLayout(parent);
    m_Layout->setContentsMargins(0,0,0,0);
    m_Layout->setSpacing(0);
    m_Layout->setRowStretch(0, 0);
    m_Layout->setRowStretch(1, 10);
    m_Layout->setRowStretch(2, 0);
    m_Layout->setRowStretch(3, 0);

    m_ContainerForControlsWidget = new QWidget(parent);
    m_MorphologicalControls = new MIDASMorphologicalSegmentorViewControlsImpl();
    m_MorphologicalControls->setupUi(m_ContainerForControlsWidget);
    m_MorphologicalControls->m_TabWidget->setCurrentIndex(0);

    QmitkMIDASBaseSegmentationFunctionality::CreateQtPartControl(parent);

    m_Layout->setMargin(9);
    m_Layout->setSpacing(6);

    m_Layout->addWidget(m_ContainerForSelectorWidget,         0, 0);
    m_Layout->addWidget(m_ContainerForSegmentationViewWidget, 1, 0);
    m_Layout->addWidget(m_ContainerForToolWidget,             2, 0);
    m_Layout->addWidget(m_ContainerForControlsWidget,         3, 0);

    m_ToolSelector->m_ManualToolSelectionBox->SetDisplayedToolGroups("Paintbrush");

    m_PipelineManager = mitk::MIDASMorphologicalSegmentorPipelineManager::New();
    m_PipelineManager->SetDataStorage(this->GetDataStorage());
    m_PipelineManager->SetToolManager(this->GetToolManager());

    this->CreateConnections();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::SetFocus()
{
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::CreateConnections()
{
  QmitkMIDASBaseSegmentationFunctionality::CreateConnections();

  if (m_MorphologicalControls != NULL)
  {
    connect(m_ImageAndSegmentationSelector->m_NewSegmentationButton, SIGNAL(released()), this, SLOT(OnCreateNewSegmentationButtonPressed()) );
    connect(m_ToolSelector, SIGNAL(ToolSelected(int)), this, SLOT(OnToolSelected(int)));
    connect(m_MorphologicalControls, SIGNAL(ThresholdingValuesChanged(double, double, int)), this, SLOT(OnThresholdingValuesChanged(double, double, int)));
    connect(m_MorphologicalControls, SIGNAL(ErosionsValuesChanged(double, int)), this, SLOT(OnErosionsValuesChanged(double, int)));
    connect(m_MorphologicalControls, SIGNAL(DilationValuesChanged(double, double, int)), this, SLOT(OnDilationValuesChanged(double, double, int)));
    connect(m_MorphologicalControls, SIGNAL(RethresholdingValuesChanged(int)), this, SLOT(OnRethresholdingValuesChanged(int)));
    connect(m_MorphologicalControls, SIGNAL(TabChanged(int)), this, SLOT(OnTabChanged(int)));
    connect(m_MorphologicalControls, SIGNAL(OKButtonClicked()), this, SLOT(OnOKButtonClicked()));
//    connect(m_MorphologicalControls, SIGNAL(CancelButtonClicked()), this, SLOT(OnCancelButtonClicked()));
    connect(m_MorphologicalControls, SIGNAL(RestartButtonClicked()), this, SLOT(OnRestartButtonClicked()));
  }
}


//-----------------------------------------------------------------------------
bool MIDASMorphologicalSegmentorView::IsNodeASegmentationImage(const mitk::DataNode::Pointer node)
{
  return m_PipelineManager->IsNodeASegmentationImage(node);
}


//-----------------------------------------------------------------------------
bool MIDASMorphologicalSegmentorView::IsNodeAWorkingImage(const mitk::DataNode::Pointer node)
{
  return m_PipelineManager->IsNodeAWorkingImage(node);
}


//-----------------------------------------------------------------------------
bool MIDASMorphologicalSegmentorView::CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node)
{
  return m_PipelineManager->CanStartSegmentationForBinaryNode(node);
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType MIDASMorphologicalSegmentorView::GetWorkingNodesFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  return m_PipelineManager->GetWorkingNodesFromSegmentationNode(node);
}


//-----------------------------------------------------------------------------
mitk::DataNode* MIDASMorphologicalSegmentorView::GetSegmentationNodeFromWorkingNode(const mitk::DataNode::Pointer node)
{
  return m_PipelineManager->GetSegmentationNodeFromWorkingNode(node);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::EnableSegmentationWidgets(bool b)
{
  int tabNumber = m_MorphologicalControls->GetTabNumber();
  if (b && (tabNumber == 1 || tabNumber == 2))
  {
    m_ToolSelector->SetEnabled(true);
  }
  else
  {
    m_ToolSelector->SetEnabled(false);
  }

  m_MorphologicalControls->EnableControls(b);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::NodeChanged(const mitk::DataNode* node)
{
  m_PipelineManager->NodeChanged(node);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::NodeRemoved(const mitk::DataNode* removedNode)
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNodeFromToolManager();
  if (segmentationNode.IsNotNull() && segmentationNode.GetPointer() == removedNode)
  {
    this->OnToolSelected(-1);
    this->EnableSegmentationWidgets(false);
    m_MorphologicalControls->m_TabWidget->blockSignals(true);
    m_MorphologicalControls->m_TabWidget->setCurrentIndex(0);
    m_MorphologicalControls->m_TabWidget->blockSignals(false);
    m_PipelineManager->RemoveWorkingData();
    m_PipelineManager->DestroyPipeline();
//    this->GetDataStorage()->Remove(segmentationNode);
    this->FireNodeSelected(this->GetReferenceNodeFromToolManager());
    this->RequestRenderWindowUpdate();
    mitk::UndoController::GetCurrentUndoModel()->Clear();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes)
{
  QmitkMIDASBaseSegmentationFunctionality::OnSelectionChanged(part, nodes);

  bool enableWidgets = false;

  if (nodes.size() == 1)
  {
    mitk::Image::Pointer referenceImage = m_PipelineManager->GetReferenceImageFromToolManager(0);
    mitk::Image::Pointer segmentationImage = m_PipelineManager->GetSegmentationImageUsingToolManager();

    if (referenceImage.IsNotNull() && segmentationImage.IsNotNull())
    {
      this->SetControlsByParameterValues();
    }

    bool isAlreadyFinished = true;
    bool foundAlreadyFinishedProperty = nodes[0]->GetBoolProperty(mitk::MIDASMorphologicalSegmentorPipelineManager::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED.c_str(), isAlreadyFinished);

    if (foundAlreadyFinishedProperty && !isAlreadyFinished)
    {
      enableWidgets = true;
    }
  }
  this->EnableSegmentationWidgets(enableWidgets);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::SetDefaultParameterValuesFromReferenceImage()
{
  m_PipelineManager->SetDefaultParameterValuesFromReferenceImage();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::SetControlsByImageData()
{
  mitk::Image::Pointer image = m_PipelineManager->GetReferenceImageFromToolManager(0);
  if (image.IsNotNull())
  {
    int axialAxis = this->GetReferenceImageAxialAxis();
    int numberOfAxialSlices = image->GetDimension(axialAxis);
    int upDirection = mitk::GetUpDirection(image, MIDAS_ORIENTATION_AXIAL);

    m_MorphologicalControls->SetControlsByImageData(    
        image->GetStatistics()->GetScalarValueMin(),
        image->GetStatistics()->GetScalarValueMax(),
        numberOfAxialSlices,
        upDirection);
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::SetControlsByParameterValues()
{
  this->SetControlsByImageData();

  MorphologicalSegmentorPipelineParams params;
  m_PipelineManager->GetParameterValuesFromSegmentationNode(params);

  m_MorphologicalControls->SetControlsByParameterValues(params);
}
