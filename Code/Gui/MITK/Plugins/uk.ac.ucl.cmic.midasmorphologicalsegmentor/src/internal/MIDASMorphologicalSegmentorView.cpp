/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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

#include "itkConversionUtils.h"
#include "mitkITKRegionParametersDataNodeProperty.h"
#include "mitkMIDASTool.h"
#include "mitkMIDASPaintbrushTool.h"
#include "QmitkMIDASMultiViewWidget.h"

const std::string MIDASMorphologicalSegmentorView::VIEW_ID = "uk.ac.ucl.cmic.midasmorphologicalsegmentor";

//-----------------------------------------------------------------------------
MIDASMorphologicalSegmentorView::MIDASMorphologicalSegmentorView()
: QmitkMIDASBaseSegmentationFunctionality()
, m_Layout(NULL)
, m_ContainerForControlsWidget(NULL)
, m_MorphologicalControls(NULL)
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
  if (this->m_PipelineManager->HasSegmentationNode())
  {
    this->OnCancelButtonClicked();
  }
}


//-----------------------------------------------------------------------------
mitk::DataNode* MIDASMorphologicalSegmentorView::OnCreateNewSegmentationButtonPressed()
{
  // Create the new segmentation, either using a previously selected one, or create a new volume.
  mitk::DataNode::Pointer newSegmentation = NULL;
  bool isRestarting = false;

  // Make sure we have a reference images... which should always be true at this point.
  mitk::Image::Pointer image = this->m_PipelineManager->GetReferenceImageFromToolManager(0);
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
      newSegmentation = QmitkMIDASBaseSegmentationFunctionality::OnCreateNewSegmentationButtonPressed(m_DefaultSegmentationColor);

      // The above method returns NULL if the user exited the colour selection dialog box.
      if (newSegmentation.IsNull())
      {
        return NULL;
      }
    }

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
      mitk::DataNode::Pointer erodeSubtractNode = paintbrushTool->CreateEmptySegmentationNode( image, mitk::MIDASTool::MORPH_EDITS_EROSIONS_SUBTRACTIONS, col->GetColor());
      erodeSubtractNode->SetBoolProperty("helper object", true);
      erodeSubtractNode->SetColor(col->GetColor());
      erodeSubtractNode->SetProperty("binaryimage.selectedcolor", col);

      mitk::DataNode::Pointer erodeAddNode = paintbrushTool->CreateEmptySegmentationNode( image, mitk::MIDASTool::MORPH_EDITS_EROSIONS_ADDITIONS, col->GetColor());
      erodeAddNode->SetBoolProperty("helper object", true);
      erodeAddNode->SetBoolProperty("visible", false);
      erodeAddNode->SetColor(segCol);
      erodeAddNode->SetProperty("binaryimage.selectedcolor", segmentationColor);

      mitk::DataNode::Pointer dilateSubtractNode = paintbrushTool->CreateEmptySegmentationNode( image, mitk::MIDASTool::MORPH_EDITS_DILATIONS_SUBTRACTIONS, col->GetColor());
      dilateSubtractNode->SetBoolProperty("helper object", true);
      dilateSubtractNode->SetColor(col->GetColor());
      dilateSubtractNode->SetProperty("binaryimage.selectedcolor", col);

      mitk::DataNode::Pointer dilateAddNode = paintbrushTool->CreateEmptySegmentationNode( image, mitk::MIDASTool::MORPH_EDITS_DILATIONS_ADDITIONS, col->GetColor());
      dilateAddNode->SetBoolProperty("helper object", true);
      dilateAddNode->SetBoolProperty("visible", false);
      dilateAddNode->SetColor(segCol);
      dilateAddNode->SetProperty("binaryimage.selectedcolor", segmentationColor);

      this->ApplyDisplayOptions(erodeSubtractNode);
      this->ApplyDisplayOptions(erodeAddNode);
      this->ApplyDisplayOptions(dilateSubtractNode);
      this->ApplyDisplayOptions(dilateAddNode);

      // Add the image to data storage, and specify this derived image as the one the toolManager will edit to.
      this->GetDataStorage()->Add(erodeSubtractNode, newSegmentation); // add as a child, because the segmentation "derives" from the original
      this->GetDataStorage()->Add(erodeAddNode, newSegmentation); // add as a child, because the segmentation "derives" from the original
      this->GetDataStorage()->Add(dilateSubtractNode, newSegmentation); // add as a child, because the segmentation "derives" from the original
      this->GetDataStorage()->Add(dilateAddNode, newSegmentation); // add as a child, because the segmentation "derives" from the original

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
      this->RequestRenderWindowUpdate();
    }
    catch (std::bad_alloc)
    {
      QMessageBox::warning(NULL,"Create new segmentation","Could not allocate memory for new segmentation");
    }

  } // end if we have a reference image

  // And... relax.
  return newSegmentation;
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnThresholdingValuesChanged(double lowerThreshold, double upperThreshold, int axialSliceNumber)
{
  this->m_PipelineManager->OnThresholdingValuesChanged(lowerThreshold, upperThreshold, axialSliceNumber);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnErosionsValuesChanged(double upperThreshold, int numberOfErosions)
{
  this->m_PipelineManager->OnErosionsValuesChanged(upperThreshold, numberOfErosions);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnDilationValuesChanged(double lowerPercentage, double upperPercentage, int numberOfDilations)
{
  this->m_PipelineManager->OnDilationValuesChanged(lowerPercentage, upperPercentage, numberOfDilations);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnRethresholdingValuesChanged(int boxSize)
{
  this->m_PipelineManager->OnRethresholdingValuesChanged(boxSize);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnTabChanged(int i)
{
  mitk::DataNode::Pointer segmentationNode = this->m_PipelineManager->GetSegmentationNodeFromToolManager();
  if (segmentationNode.IsNotNull())
  {
    if (i == 1 || i == 2)
    {
      this->m_ToolSelector->SetEnabled(true);

      mitk::ToolManager::Pointer toolManager = this->GetToolManager();
      mitk::MIDASPaintbrushTool::Pointer paintbrushTool = dynamic_cast<mitk::MIDASPaintbrushTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASPaintbrushTool>()));

      mitk::DataNode::Pointer erodeSubtractNode = this->GetToolManager()->GetWorkingData(1);
      mitk::DataNode::Pointer dilateSubtractNode = this->GetToolManager()->GetWorkingData(3);

      if (i == 1)
      {
        paintbrushTool->SetErosionMode(true);
        erodeSubtractNode->SetVisibility(true);
        dilateSubtractNode->SetVisibility(false);
      }
      else
      {
        paintbrushTool->SetErosionMode(false);
        erodeSubtractNode->SetVisibility(true);
        dilateSubtractNode->SetVisibility(false);
      }
    }
    else
    {
      this->m_ToolSelector->SetEnabled(false);
      this->OnToolSelected(-1); // make sure we de-activate tools.
    }

    segmentationNode->SetIntProperty("midas.morph.stage", i);
    this->m_PipelineManager->UpdateSegmentation();
    this->RequestRenderWindowUpdate();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnOKButtonClicked()
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNodeFromToolManager();
  if (segmentationNode.IsNotNull())
  {
    this->OnToolSelected(-1);
    this->EnableSegmentationWidgets(false);
    this->m_MorphologicalControls->m_TabWidget->blockSignals(true);
    this->m_MorphologicalControls->m_TabWidget->setCurrentIndex(0);
    this->m_MorphologicalControls->m_TabWidget->blockSignals(false);
    this->m_PipelineManager->FinalizeSegmentation();
    this->FireNodeSelected(this->GetReferenceNodeFromToolManager());
    this->RequestRenderWindowUpdate();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnClearButtonClicked()
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNodeFromToolManager();
  if (segmentationNode.IsNotNull())
  {
    this->OnToolSelected(-1);
    this->m_PipelineManager->ClearWorkingData();
    this->SetDefaultParameterValuesFromReferenceImage();
    this->SetControlsByImageData();
    this->SetControlsByParameterValues();
    this->m_PipelineManager->UpdateSegmentation();
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
    this->m_MorphologicalControls->m_TabWidget->blockSignals(true);
    this->m_MorphologicalControls->m_TabWidget->setCurrentIndex(0);
    this->m_MorphologicalControls->m_TabWidget->blockSignals(false);
    this->m_PipelineManager->ClearWorkingData();
    this->m_PipelineManager->DestroyPipeline();
    this->m_PipelineManager->RemoveWorkingData();
    this->GetDataStorage()->Remove(segmentationNode);
    this->FireNodeSelected(this->GetReferenceNodeFromToolManager());
    this->RequestRenderWindowUpdate();
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
    connect(m_MorphologicalControls, SIGNAL(CancelButtonClicked()), this, SLOT(OnCancelButtonClicked()));
    connect(m_MorphologicalControls, SIGNAL(ClearButtonClicked()), this, SLOT(OnClearButtonClicked()));
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
  this->m_MorphologicalControls->EnableControls(b);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::NodeChanged(const mitk::DataNode* node)
{
  this->m_PipelineManager->NodeChanged(node);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes)
{
  QmitkMIDASBaseSegmentationFunctionality::OnSelectionChanged(part, nodes);

  bool enableWidgets = false;

  if (nodes.size() == 1)
  {
    mitk::Image::Pointer referenceImage = this->m_PipelineManager->GetReferenceImageFromToolManager(0);
    mitk::Image::Pointer segmentationImage = this->m_PipelineManager->GetSegmentationImageUsingToolManager();

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
  this->m_PipelineManager->SetDefaultParameterValuesFromReferenceImage();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::SetControlsByImageData()
{
  mitk::Image::Pointer image = this->m_PipelineManager->GetReferenceImageFromToolManager(0);
  if (image.IsNotNull())
  {
    int axialAxis = this->GetReferenceImageAxialAxis();
    int numberOfAxialSlices = image->GetDimension(axialAxis);

    m_MorphologicalControls->SetControlsByImageData(
        image->GetStatistics()->GetScalarValueMin(),
        image->GetStatistics()->GetScalarValueMax(),
        numberOfAxialSlices);
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorView::SetControlsByParameterValues()
{
  this->SetControlsByImageData();

  MorphologicalSegmentorPipelineParams params;
  this->m_PipelineManager->GetParameterValuesFromSegmentationNode(params);

  this->m_MorphologicalControls->SetControlsByParameterValues(params);
}
