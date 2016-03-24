/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMorphologicalSegmentorView.h"

#include <QMessageBox>

#include <berryIWorkbenchPage.h>

#include <mitkColorProperty.h>
#include <mitkDataStorageUtils.h>
#include <mitkImage.h>
#include <mitkImageAccessByItk.h>
#include <mitkImageCast.h>
#include <mitkImageStatisticsHolder.h>
#include <mitkITKImageImport.h>
#include <mitkPlane.h>
#include <mitkUndoController.h>

#include <niftkMIDASImageUtils.h>
#include <niftkMIDASOrientationUtils.h>

#include <itkConversionUtils.h>
#include <mitkITKRegionParametersDataNodeProperty.h>
#include <niftkMIDASTool.h>
#include <niftkMIDASPaintbrushTool.h>

#include <niftkMIDASOrientationUtils.h>

#include <niftkMorphologicalSegmentorControls.h>


const std::string niftkMorphologicalSegmentorView::VIEW_ID = "uk.ac.ucl.cmic.midasmorphologicalsegmentor";

//-----------------------------------------------------------------------------
niftkMorphologicalSegmentorView::niftkMorphologicalSegmentorView()
: niftkBaseSegmentorView()
, m_MorphologicalSegmentorControls(NULL)
, m_PipelineManager(NULL)
, m_TabIndex(-1)
{
}


//-----------------------------------------------------------------------------
niftkMorphologicalSegmentorView::niftkMorphologicalSegmentorView(
    const niftkMorphologicalSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
niftkMorphologicalSegmentorView::~niftkMorphologicalSegmentorView()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  int paintbrushToolId = toolManager->GetToolIdByToolType<niftk::MIDASPaintbrushTool>();
  niftk::MIDASPaintbrushTool* paintbrushTool = dynamic_cast<niftk::MIDASPaintbrushTool*>(toolManager->GetToolById(paintbrushToolId));
  assert(paintbrushTool);

  paintbrushTool->SegmentationEdited.RemoveListener(mitk::MessageDelegate1<niftkMorphologicalSegmentorView, int>(this, &niftkMorphologicalSegmentorView::OnSegmentationEdited));
}


//-----------------------------------------------------------------------------
std::string niftkMorphologicalSegmentorView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::ClosePart()
{
  if (m_PipelineManager->HasSegmentationNode())
  {
    this->OnCancelButtonClicked();
  }
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::RegisterTools(mitk::ToolManager::Pointer toolManager)
{
  toolManager->RegisterTool("MIDASPaintbrushTool");

  int paintbrushToolId = toolManager->GetToolIdByToolType<niftk::MIDASPaintbrushTool>();
  niftk::MIDASPaintbrushTool* paintbrushTool = dynamic_cast<niftk::MIDASPaintbrushTool*>(toolManager->GetToolById(paintbrushToolId));
  assert(paintbrushTool);

  paintbrushTool->SegmentationEdited.AddListener(mitk::MessageDelegate1<niftkMorphologicalSegmentorView, int>(this, &niftkMorphologicalSegmentorView::OnSegmentationEdited));
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::OnCreateNewSegmentationButtonPressed()
{
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

    int paintbrushToolId = toolManager->GetToolIdByToolType<niftk::MIDASPaintbrushTool>();

    mitk::Tool* paintbrushTool = toolManager->GetToolById(paintbrushToolId);
    assert(paintbrushTool);

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

    mitk::DataNode::Pointer axialCutOffPlaneNode = this->CreateAxialCutOffPlaneNode(image);
    this->GetDataStorage()->Add(axialCutOffPlaneNode, newSegmentation);

    this->WaitCursorOn();

    // Mark the newSegmentation as "unfinished".
    newSegmentation->SetBoolProperty(niftk::MorphologicalSegmentorPipelineManager::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED.c_str(), false);

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
      mitk::DataNode::Pointer erodeAddNode = paintbrushTool->CreateEmptySegmentationNode(image, niftk::MIDASPaintbrushTool::EROSIONS_ADDITIONS_NAME, col->GetColor());
      erodeAddNode->SetBoolProperty("helper object", true);
      erodeAddNode->SetBoolProperty("visible", false);
      erodeAddNode->SetColor(segCol);
      erodeAddNode->SetProperty("binaryimage.selectedcolor", segmentationColor);
      erodeAddNode->AddProperty(niftk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str(), erodeAddEditingProp);

      mitk::ITKRegionParametersDataNodeProperty::Pointer erodeSubtractEditingProp = mitk::ITKRegionParametersDataNodeProperty::New();
      erodeSubtractEditingProp->SetSize(1,1,1);
      erodeSubtractEditingProp->SetValid(false);
      mitk::DataNode::Pointer erodeSubtractNode = paintbrushTool->CreateEmptySegmentationNode( image, niftk::MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS_NAME, col->GetColor());
      erodeSubtractNode->SetBoolProperty("helper object", true);
      erodeSubtractNode->SetBoolProperty("visible", false);
      erodeSubtractNode->SetColor(col->GetColor());
      erodeSubtractNode->SetProperty("binaryimage.selectedcolor", col);
      erodeSubtractNode->AddProperty(niftk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str(), erodeSubtractEditingProp);

      mitk::ITKRegionParametersDataNodeProperty::Pointer dilateAddEditingProp = mitk::ITKRegionParametersDataNodeProperty::New();
      dilateAddEditingProp->SetSize(1,1,1);
      dilateAddEditingProp->SetValid(false);
      mitk::DataNode::Pointer dilateAddNode = paintbrushTool->CreateEmptySegmentationNode( image, niftk::MIDASPaintbrushTool::DILATIONS_ADDITIONS_NAME, col->GetColor());
      dilateAddNode->SetBoolProperty("helper object", true);
      dilateAddNode->SetBoolProperty("visible", false);
      dilateAddNode->SetColor(segCol);
      dilateAddNode->SetProperty("binaryimage.selectedcolor", segmentationColor);
      dilateAddNode->AddProperty(niftk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str(), dilateAddEditingProp);

      mitk::ITKRegionParametersDataNodeProperty::Pointer dilateSubtractEditingProp = mitk::ITKRegionParametersDataNodeProperty::New();
      dilateSubtractEditingProp->SetSize(1,1,1);
      dilateSubtractEditingProp->SetValid(false);
      mitk::DataNode::Pointer dilateSubtractNode = paintbrushTool->CreateEmptySegmentationNode( image, niftk::MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS_NAME, col->GetColor());
      dilateSubtractNode->SetBoolProperty("helper object", true);
      dilateSubtractNode->SetBoolProperty("visible", false);
      dilateSubtractNode->SetColor(col->GetColor());
      dilateSubtractNode->SetProperty("binaryimage.selectedcolor", col);
      dilateSubtractNode->AddProperty(niftk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str(), dilateSubtractEditingProp);

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
      workingData[niftk::MIDASPaintbrushTool::EROSIONS_ADDITIONS] = erodeAddNode;
      workingData[niftk::MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS] = erodeSubtractNode;
      workingData[niftk::MIDASPaintbrushTool::DILATIONS_ADDITIONS] = dilateAddNode;
      workingData[niftk::MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS] = dilateSubtractNode;

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

    this->FocusOnCurrentWindow();
    this->RequestRenderWindowUpdate();
    this->WaitCursorOff();

  } // end if we have a reference image

  // Finally, select the new segmentation node.
  this->SetCurrentSelection(newSegmentation);
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer niftkMorphologicalSegmentorView::CreateAxialCutOffPlaneNode(const mitk::Image* referenceImage)
{
  mitk::BaseGeometry* geometry = referenceImage->GetGeometry();

  int axialAxis = niftk::GetThroughPlaneAxis(referenceImage, MIDAS_ORIENTATION_AXIAL);
  int sagittalAxis = niftk::GetThroughPlaneAxis(referenceImage, MIDAS_ORIENTATION_SAGITTAL);
  int coronalAxis = niftk::GetThroughPlaneAxis(referenceImage, MIDAS_ORIENTATION_CORONAL);

  int axialUpDirection = niftk::GetUpDirection(referenceImage, MIDAS_ORIENTATION_AXIAL);
  int sagittalUpDirection = niftk::GetUpDirection(referenceImage, MIDAS_ORIENTATION_SAGITTAL);
  int coronalUpDirection = niftk::GetUpDirection(referenceImage, MIDAS_ORIENTATION_CORONAL);

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
void niftkMorphologicalSegmentorView::OnThresholdingValuesChanged(double lowerThreshold, double upperThreshold, int axialSliceNumber)
{
  m_PipelineManager->OnThresholdingValuesChanged(lowerThreshold, upperThreshold, axialSliceNumber);

  mitk::DataNode::Pointer referenceImageNode = this->GetReferenceNodeFromToolManager();
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNode();
  mitk::Image* referenceImage = dynamic_cast<mitk::Image*>(referenceImageNode->GetData());
  mitk::BaseGeometry* geometry = referenceImage->GetGeometry();

  int axialAxis = niftk::GetThroughPlaneAxis(referenceImage, MIDAS_ORIENTATION_AXIAL);
  int axialUpDirection = niftk::GetUpDirection(referenceImage, MIDAS_ORIENTATION_AXIAL);

  mitk::Plane* axialCutOffPlane = this->GetDataStorage()->GetNamedDerivedObject<mitk::Plane>("Axial cut-off plane", segmentationNode);

  // Lift the axial cut-off plane to the height determined by axialSliceNumber.
  mitk::Point3D planeCentre = axialCutOffPlane->GetGeometry()->GetOrigin();
  planeCentre[2] = geometry->GetOrigin()[2] + (axialUpDirection * axialSliceNumber - 0.5) * geometry->GetSpacing()[axialAxis];
  axialCutOffPlane->SetOrigin(planeCentre);

  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::OnErosionsValuesChanged(double upperThreshold, int numberOfErosions)
{
  m_PipelineManager->OnErosionsValuesChanged(upperThreshold, numberOfErosions);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::OnDilationsValuesChanged(double lowerPercentage, double upperPercentage, int numberOfDilations)
{
  m_PipelineManager->OnDilationsValuesChanged(lowerPercentage, upperPercentage, numberOfDilations);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::OnRethresholdingValuesChanged(int boxSize)
{
  m_PipelineManager->OnRethresholdingValuesChanged(boxSize);
  this->RequestRenderWindowUpdate();
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::OnTabChanged(int tabIndex)
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNode();
  if (segmentationNode.IsNotNull())
  {
    if (tabIndex == 1 || tabIndex == 2)
    {
      m_MorphologicalSegmentorControls->m_ToolSelectorWidget->SetEnabled(true);

      mitk::ToolManager::Pointer toolManager = this->GetToolManager();
      niftk::MIDASPaintbrushTool::Pointer paintbrushTool = dynamic_cast<niftk::MIDASPaintbrushTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<niftk::MIDASPaintbrushTool>()));

      mitk::DataNode::Pointer erodeSubtractNode = this->GetToolManager()->GetWorkingData(niftk::MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS);
      mitk::DataNode::Pointer dilateSubtractNode = this->GetToolManager()->GetWorkingData(niftk::MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS);

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
      m_MorphologicalSegmentorControls->m_ToolSelectorWidget->SetEnabled(false);
      this->OnToolSelected(-1); // make sure we de-activate tools.
    }

    m_PipelineManager->OnTabChanged(tabIndex);

    this->RequestRenderWindowUpdate();
  }

  m_TabIndex = tabIndex;
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::OnOKButtonClicked()
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNode();
  if (segmentationNode.IsNotNull())
  {
    this->OnToolSelected(-1);
    this->EnableSegmentationWidgets(false);
    bool wasBlocked = m_MorphologicalSegmentorControls->m_TabWidget->blockSignals(true);
    m_MorphologicalSegmentorControls->m_TabWidget->setCurrentIndex(0);
    m_MorphologicalSegmentorControls->m_TabWidget->blockSignals(wasBlocked);
    m_PipelineManager->FinalizeSegmentation();

    /// Remove the axial cut-off plane node from the data storage.
    mitk::DataNode::Pointer axialCutOffPlaneNode = this->GetDataStorage()->GetNamedDerivedNode("Axial cut-off plane", segmentationNode);
    this->GetDataStorage()->Remove(axialCutOffPlaneNode);

    this->FireNodeSelected(this->GetReferenceNodeFromToolManager());
    this->RequestRenderWindowUpdate();
    mitk::UndoController::GetCurrentUndoModel()->Clear();
  }
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::OnRestartButtonClicked()
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
      mitk::DataNode::Pointer referenceImageNode = this->GetReferenceNodeFromToolManager();
      mitk::Image* referenceImage = dynamic_cast<mitk::Image*>(referenceImageNode->GetData());
      mitk::BaseGeometry* geometry = referenceImage->GetGeometry();

      mitk::Plane* axialCutOffPlane = this->GetDataStorage()->GetNamedDerivedObject<mitk::Plane>("Axial cut-off plane", segmentationNode);

      int axialAxis = niftk::GetThroughPlaneAxis(referenceImage, MIDAS_ORIENTATION_AXIAL);

      // The centre of the plane is the same as the centre of the image, but it is shifted
      // along the axial axis to a position determined by axialSliceNumber.
      // As an initial point we set it one slice below the 'height' of the origin.
      mitk::Point3D planeCentre = geometry->GetCenter();
      planeCentre[2] = geometry->GetOrigin()[axialAxis] - geometry->GetSpacing()[axialAxis];

      axialCutOffPlane->SetOrigin(planeCentre);
    }

    this->FireNodeSelected(segmentationNode);
    this->RequestRenderWindowUpdate();
  }
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::OnCancelButtonClicked()
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNode();
  if (segmentationNode.IsNotNull())
  {
    this->OnToolSelected(-1);
    this->EnableSegmentationWidgets(false);
    bool wasBlocked = m_MorphologicalSegmentorControls->m_TabWidget->blockSignals(true);
    m_MorphologicalSegmentorControls->m_TabWidget->setCurrentIndex(0);
    m_MorphologicalSegmentorControls->m_TabWidget->blockSignals(wasBlocked);
    m_PipelineManager->RemoveWorkingData();
    mitk::Image::Pointer segmentationImage = dynamic_cast<mitk::Image*>(segmentationNode->GetData());
    m_PipelineManager->DestroyPipeline(segmentationImage);
    mitk::DataNode::Pointer axialCutOffPlaneNode = this->GetDataStorage()->GetNamedDerivedNode("Axial cut-off plane", segmentationNode);
    this->GetDataStorage()->Remove(axialCutOffPlaneNode);
    this->GetDataStorage()->Remove(segmentationNode);
    this->FireNodeSelected(this->GetReferenceNodeFromToolManager());
    this->RequestRenderWindowUpdate();
    mitk::UndoController::GetCurrentUndoModel()->Clear();
  }
}

//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::CreateQtPartControl(QWidget* parent)
{
  niftkBaseSegmentorView::CreateQtPartControl(parent);

  m_PipelineManager = niftk::MorphologicalSegmentorPipelineManager::New();
  m_PipelineManager->SetDataStorage(this->GetDataStorage());
  m_PipelineManager->SetToolManager(this->GetToolManager());
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorControls* niftkMorphologicalSegmentorView::CreateSegmentorControls(QWidget *parent)
{
  m_MorphologicalSegmentorControls = new niftkMorphologicalSegmentorControls(parent);
  return m_MorphologicalSegmentorControls;
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::SetFocus()
{
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::CreateConnections()
{
  niftkBaseSegmentorView::CreateConnections();

  this->connect(m_MorphologicalSegmentorControls->m_SegmentationSelectorWidget->m_NewSegmentationButton, SIGNAL(released()), SLOT(OnCreateNewSegmentationButtonPressed()) );
  this->connect(m_MorphologicalSegmentorControls->m_ToolSelectorWidget, SIGNAL(ToolSelected(int)), SLOT(OnToolSelected(int)));
  this->connect(m_MorphologicalSegmentorControls, SIGNAL(ThresholdingValuesChanged(double, double, int)), SLOT(OnThresholdingValuesChanged(double, double, int)));
  this->connect(m_MorphologicalSegmentorControls, SIGNAL(ErosionsValuesChanged(double, int)), SLOT(OnErosionsValuesChanged(double, int)));
  this->connect(m_MorphologicalSegmentorControls, SIGNAL(DilationsValuesChanged(double, double, int)), SLOT(OnDilationsValuesChanged(double, double, int)));
  this->connect(m_MorphologicalSegmentorControls, SIGNAL(RethresholdingValuesChanged(int)), SLOT(OnRethresholdingValuesChanged(int)));
  this->connect(m_MorphologicalSegmentorControls, SIGNAL(TabChanged(int)), SLOT(OnTabChanged(int)));
  this->connect(m_MorphologicalSegmentorControls, SIGNAL(OKButtonClicked()), SLOT(OnOKButtonClicked()));
//  this->connect(m_MorphologicalControls, SIGNAL(CancelButtonClicked()), SLOT(OnCancelButtonClicked()));
  this->connect(m_MorphologicalSegmentorControls, SIGNAL(RestartButtonClicked()), SLOT(OnRestartButtonClicked()));
}


//-----------------------------------------------------------------------------
bool niftkMorphologicalSegmentorView::IsNodeASegmentationImage(const mitk::DataNode::Pointer node)
{
  return m_PipelineManager->IsNodeASegmentationImage(node);
}


//-----------------------------------------------------------------------------
bool niftkMorphologicalSegmentorView::IsNodeAWorkingImage(const mitk::DataNode::Pointer node)
{
  return m_PipelineManager->IsNodeAWorkingImage(node);
}


//-----------------------------------------------------------------------------
bool niftkMorphologicalSegmentorView::CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node)
{
  return m_PipelineManager->CanStartSegmentationForBinaryNode(node);
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType niftkMorphologicalSegmentorView::GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  return m_PipelineManager->GetWorkingDataFromSegmentationNode(node);
}


//-----------------------------------------------------------------------------
mitk::DataNode* niftkMorphologicalSegmentorView::GetSegmentationNodeFromWorkingData(const mitk::DataNode::Pointer node)
{
  return m_PipelineManager->GetSegmentationNodeFromWorkingData(node);
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::EnableSegmentationWidgets(bool enabled)
{
  int tabIndex = m_MorphologicalSegmentorControls->GetTabIndex();
  if (enabled && (tabIndex == 1 || tabIndex == 2))
  {
    m_MorphologicalSegmentorControls->m_ToolSelectorWidget->SetEnabled(true);
  }
  else
  {
    m_MorphologicalSegmentorControls->m_ToolSelectorWidget->SetEnabled(false);
  }

  m_MorphologicalSegmentorControls->SetEnabled(enabled);
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::OnSegmentationEdited(int imageIndex)
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  if (toolManager)
  {
    mitk::DataNode* node = toolManager->GetWorkingData(imageIndex);
    assert(node);
    mitk::ITKRegionParametersDataNodeProperty::Pointer prop =
        dynamic_cast<mitk::ITKRegionParametersDataNodeProperty*>(node->GetProperty(niftk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str()));
    if (prop.IsNotNull() && prop->HasVolume())
    {
      m_PipelineManager->UpdateSegmentation();
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::NodeRemoved(const mitk::DataNode* removedNode)
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNode();
  if (segmentationNode.IsNotNull() && segmentationNode.GetPointer() == removedNode)
  {
    this->OnToolSelected(-1);
    this->EnableSegmentationWidgets(false);
    bool wasBlocked = m_MorphologicalSegmentorControls->m_TabWidget->blockSignals(true);
    m_MorphologicalSegmentorControls->m_TabWidget->setCurrentIndex(0);
    m_MorphologicalSegmentorControls->m_TabWidget->blockSignals(wasBlocked);
    m_PipelineManager->RemoveWorkingData();
    mitk::Image::Pointer segmentationImage = dynamic_cast<mitk::Image*>(segmentationNode->GetData());
    m_PipelineManager->DestroyPipeline(segmentationImage);
//    this->GetDataStorage()->Remove(segmentationNode);
    this->FireNodeSelected(this->GetReferenceNodeFromToolManager());
    this->RequestRenderWindowUpdate();
    mitk::UndoController::GetCurrentUndoModel()->Clear();
  }
}


//-----------------------------------------------------------------------------
QString niftkMorphologicalSegmentorView::GetPreferencesNodeName()
{
  return niftkMorphologicalSegmentorPreferencePage::PREFERENCES_NODE_NAME;
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes)
{
  niftkBaseSegmentorView::OnSelectionChanged(part, nodes);

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
    bool foundAlreadyFinishedProperty = nodes[0]->GetBoolProperty(niftk::MorphologicalSegmentorPipelineManager::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED.c_str(), isAlreadyFinished);

    if (foundAlreadyFinishedProperty && !isAlreadyFinished)
    {
      enableWidgets = true;
    }
  }
  this->EnableSegmentationWidgets(enableWidgets);
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::SetSegmentationNodePropsFromReferenceImage()
{
  m_PipelineManager->SetSegmentationNodePropsFromReferenceImage();
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::SetControlsFromReferenceImage()
{
  mitk::Image::ConstPointer referenceImage = m_PipelineManager->GetReferenceImage();
  if (referenceImage.IsNotNull())
  {
    int axialAxis = this->GetReferenceImageAxialAxis();
    int numberOfAxialSlices = referenceImage->GetDimension(axialAxis);
    int upDirection = niftk::GetUpDirection(referenceImage, MIDAS_ORIENTATION_AXIAL);

    m_MorphologicalSegmentorControls->SetControlsByReferenceImage(
        referenceImage->GetStatistics()->GetScalarValueMin(),
        referenceImage->GetStatistics()->GetScalarValueMax(),
        numberOfAxialSlices,
        upDirection);
  }
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::SetControlsFromSegmentationNodeProps()
{
  MorphologicalSegmentorPipelineParams params;
  m_PipelineManager->GetPipelineParamsFromSegmentationNode(params);

  m_MorphologicalSegmentorControls->SetControlsByPipelineParams(params);
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::onVisibilityChanged(const mitk::DataNode* node)
{
  mitk::DataNode::Pointer segmentationNode = m_PipelineManager->GetSegmentationNode();

  std::vector<mitk::DataNode*> workingData = this->GetWorkingData();
  if (segmentationNode.IsNotNull() && node == segmentationNode && workingData.size() == 4)
  {
    mitk::DataNode::Pointer axialCutOffPlaneNode = this->GetDataStorage()->GetNamedDerivedNode("Axial cut-off plane", segmentationNode);

    bool segmentationNodeVisibility;
    if (node->GetVisibility(segmentationNodeVisibility, 0) && segmentationNodeVisibility)
    {
      workingData[niftk::MIDASPaintbrushTool::EROSIONS_ADDITIONS]->SetVisibility(false);
      workingData[niftk::MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS]->SetVisibility(m_MorphologicalSegmentorControls->m_TabWidget->currentIndex() == 1);
      workingData[niftk::MIDASPaintbrushTool::DILATIONS_ADDITIONS]->SetVisibility(false);
      workingData[niftk::MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS]->SetVisibility(m_MorphologicalSegmentorControls->m_TabWidget->currentIndex() == 2);
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
