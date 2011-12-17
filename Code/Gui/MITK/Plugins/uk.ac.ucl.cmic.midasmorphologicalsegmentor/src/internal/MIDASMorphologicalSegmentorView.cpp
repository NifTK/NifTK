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

#include "mitkImageAccessByItk.h"
#include "mitkITKImageImport.h"
#include "mitkRenderingManager.h"
#include "mitkImageCast.h"
#include "mitkImage.h"
#include "mitkMIDASTool.h"
#include "mitkMIDASPaintbrushTool.h"
#include "mitkColorProperty.h"
#include "mitkDataStorageUtils.h"
#include "itkImageFileWriter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkConversionUtils.h"

const std::string MIDASMorphologicalSegmentorView::VIEW_ID = "uk.ac.ucl.cmic.midasmorphologicalsegmentor";
const std::string MIDASMorphologicalSegmentorView::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED = "midas.morph.finished";

MIDASMorphologicalSegmentorView::MIDASMorphologicalSegmentorView()
: QmitkMIDASBaseSegmentationFunctionality()
, m_MorphologicalControls(NULL)
, m_Layout(NULL)
, m_ContainerForSelectorWidget(NULL)
, m_ContainerForControlsWidget(NULL)
, m_PaintbrushToolId(-1)
{
  m_ToolManager = mitk::ToolManager::New(this->GetDefaultDataStorage());
  m_ToolManager->RegisterClient();
  m_PaintbrushToolId = m_ToolManager->GetToolIdByToolType<mitk::MIDASPaintbrushTool>();
}

MIDASMorphologicalSegmentorView::MIDASMorphologicalSegmentorView(
    const MIDASMorphologicalSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

MIDASMorphologicalSegmentorView::~MIDASMorphologicalSegmentorView()
{
  if (m_MorphologicalControls != NULL)
  {
    delete m_MorphologicalControls;
  }
}

std::string MIDASMorphologicalSegmentorView::GetViewID() const
{
  return VIEW_ID;
}

void MIDASMorphologicalSegmentorView::Activated()
{
  QmitkMIDASBaseSegmentationFunctionality::Activated();
}

void MIDASMorphologicalSegmentorView::Deactivated()
{
  QmitkMIDASBaseSegmentationFunctionality::Deactivated();
}

mitk::ToolManager* MIDASMorphologicalSegmentorView::GetToolManager()
{
  return m_ToolManager;
}

bool MIDASMorphologicalSegmentorView::CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node)
{
  int tmp;
  if (node->GetIntProperty("midas.morph.stage", tmp))
  {
    return true;
  }
  else
  {
    return false;
  }
}

bool MIDASMorphologicalSegmentorView::IsNodeASegmentationImage(const mitk::DataNode::Pointer node)
{
  assert(node);
  bool result = false;

  if (IsNodeABinaryImage(node))
  {

    mitk::DataNode::Pointer parent = FindFirstParentImage(this->GetDefaultDataStorage(), node, false);

    if (parent.IsNotNull())
    {
      // Should also have two children called mitk::MIDASTool::SUBTRACTIONS_IMAGE_NAME and mitk::MIDASTool::ADDITIONS_IMAGE_NAME
      mitk::DataStorage::SetOfObjects::Pointer children = FindDerivedImages(this->GetDefaultDataStorage(), node, true);
      if (children->size() == 2)
      {
        std::string name1;
        (*children)[0]->GetStringProperty("name", name1);
        std::string name2;
        (*children)[1]->GetStringProperty("name", name2);

        if ((name1 == mitk::MIDASTool::SUBTRACTIONS_IMAGE_NAME || name1 == mitk::MIDASTool::ADDITIONS_IMAGE_NAME)
            && (name2 == mitk::MIDASTool::SUBTRACTIONS_IMAGE_NAME || name2 == mitk::MIDASTool::ADDITIONS_IMAGE_NAME)
            )
        {
          result = true;
        }
      }
    }
  }
  return result;
}

bool MIDASMorphologicalSegmentorView::IsNodeAWorkingImage(const mitk::DataNode::Pointer node)
{
  assert(node);
  bool result = false;

  if (IsNodeABinaryImage(node))
  {
    mitk::DataNode::Pointer parent = FindFirstParentImage(this->GetDefaultDataStorage(), node, true);

    if (parent.IsNotNull())
    {
      std::string name;
      if (node->GetStringProperty("name", name))
      {
        if (name == mitk::MIDASTool::SUBTRACTIONS_IMAGE_NAME || name == mitk::MIDASTool::ADDITIONS_IMAGE_NAME)
        {
          result = true;
        }
      }
    }
  }

  return result;
}

mitk::ToolManager::DataVectorType MIDASMorphologicalSegmentorView::GetWorkingNodesFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  assert(node);
  mitk::ToolManager::DataVectorType result;

  mitk::DataStorage::SetOfObjects::Pointer children = FindDerivedImages(this->GetDefaultDataStorage(), node, true );

  for (unsigned int i = 0; i < children->size(); i++)
  {
    std::string name;
    if ((*children)[i]->GetStringProperty("name", name))
    {
      if (name == mitk::MIDASTool::SUBTRACTIONS_IMAGE_NAME)
      {
        result.push_back((*children)[i]);
      }
    }
  }

  for (unsigned int i = 0; i < children->size(); i++)
  {
    std::string name;
    if ((*children)[i]->GetStringProperty("name", name))
    {
      if (name == mitk::MIDASTool::ADDITIONS_IMAGE_NAME)
      {
        result.push_back((*children)[i]);
      }
    }
  }

  if (result.size() != 2)
  {
    result.clear();
  }
  return result;
}

mitk::DataNode* MIDASMorphologicalSegmentorView::GetSegmentationNodeFromWorkingNode(const mitk::DataNode::Pointer node)
{
  assert(node);
  mitk::DataNode* result = NULL;

  if (IsNodeABinaryImage(node))
  {
    mitk::DataNode::Pointer parent = FindFirstParentImage(this->GetDefaultDataStorage(), node, true);
    if (parent.IsNotNull())
    {
      result = parent;
    }
  }

  return result;
}

mitk::DataNode* MIDASMorphologicalSegmentorView::GetSegmentationNodeUsingToolManager()
{
  mitk::DataNode *result = NULL;
  mitk::ToolManager::DataVectorType workingNodesFromToolManager = this->GetWorkingNodesFromToolManager();

  if (workingNodesFromToolManager.size() > 0)
  {
    result = FindFirstParentImage(this->GetDefaultDataStorage(), workingNodesFromToolManager[0], true);
  }
  return result;
}

mitk::Image* MIDASMorphologicalSegmentorView::GetSegmentationImageUsingToolManager()
{
  mitk::Image *result = NULL;

  mitk::DataNode *node = this->GetSegmentationNodeUsingToolManager();
  if (node != NULL)
  {
    result = static_cast<mitk::Image*>(node->GetData());
  }
  return result;
}

void MIDASMorphologicalSegmentorView::OnSelectionChanged(std::vector<mitk::DataNode*> nodes)
{
  QmitkMIDASBaseSegmentationFunctionality::OnSelectionChanged(nodes);

  bool enableWidgets = false;

  if (nodes.size() == 1)
  {
    mitk::Image* referenceImage = this->GetReferenceImageFromToolManager();
    mitk::Image* segmentationImage = this->GetSegmentationImageUsingToolManager();

    if (referenceImage != NULL && segmentationImage != NULL)
    {
      this->SetControlsByParameterValues();
    }

    bool isAlreadyFinished = true;
    bool foundAlreadyFinishedProperty = nodes[0]->GetBoolProperty(MIDASMorphologicalSegmentorView::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED.c_str(), isAlreadyFinished);

    if (foundAlreadyFinishedProperty && !isAlreadyFinished)
    {
      enableWidgets = true;
    }
  }
  this->EnableSegmentationWidgets(enableWidgets);
}

void MIDASMorphologicalSegmentorView::CreateQtPartControl(QWidget *parent)
{
  m_Parent = parent;

  if (!m_MorphologicalControls)
  {
    m_Layout = new QGridLayout(parent);

    m_ContainerForSelectorWidget = new QWidget(parent);
    m_ContainerForControlsWidget = new QWidget(parent);

    m_MorphologicalControls = new MIDASMorphologicalSegmentorViewControlsImpl();
    m_MorphologicalControls->setupUi(m_ContainerForControlsWidget);
    m_MorphologicalControls->m_TabWidget->setCurrentIndex(0);

    QmitkMIDASBaseSegmentationFunctionality::CreateQtPartControl(m_ContainerForSelectorWidget);

    m_Layout->addWidget(m_ContainerForSelectorWidget, 0, 0);
    m_Layout->addWidget(m_ContainerForControlsWidget, 1, 0);
  }
  this->CreateConnections();
}

void MIDASMorphologicalSegmentorView::CreateConnections()
{
  QmitkMIDASBaseSegmentationFunctionality::CreateConnections();

  if (m_MorphologicalControls)
  {
    connect(m_ImageAndSegmentationSelector->m_NewSegmentationButton, SIGNAL(released()), this, SLOT(OnCreateNewSegmentationButtonPressed()) );
    connect(m_MorphologicalControls, SIGNAL(ThresholdingValuesChanged(double, double, int)), this, SLOT(OnThresholdingValuesChanged(double, double, int)));
    connect(m_MorphologicalControls, SIGNAL(ErosionsValuesChanged(double, int)), this, SLOT(OnErosionsValuesChanged(double, int)));
    connect(m_MorphologicalControls, SIGNAL(DilationValuesChanged(double, double, int)), this, SLOT(OnDilationValuesChanged(double, double, int)));
    connect(m_MorphologicalControls, SIGNAL(RethresholdingValuesChanged(int)), this, SLOT(OnRethresholdingValuesChanged(int)));
    connect(m_MorphologicalControls, SIGNAL(TabChanged(int)), this, SLOT(OnTabChanged(int)));
    connect(m_MorphologicalControls, SIGNAL(CursorWidthChanged(int)), this, SLOT(OnCursorWidthChanged(int)));
    connect(m_MorphologicalControls, SIGNAL(OKButtonClicked()), this, SLOT(OnOKButtonClicked()));
    connect(m_MorphologicalControls, SIGNAL(CancelButtonClicked()), this, SLOT(OnCancelButtonClicked()));
    connect(m_MorphologicalControls, SIGNAL(ClearButtonClicked()), this, SLOT(OnClearButtonClicked()));
  }
}

void MIDASMorphologicalSegmentorView::SetDefaultParameterValuesFromReferenceImage()
{
  mitk::Image* referenceImage = this->GetReferenceImage();
  assert(referenceImage);

  mitk::DataNode* segmentationNode = this->GetSegmentationNodeUsingToolManager();
  assert(segmentationNode);

  segmentationNode->SetIntProperty("midas.morph.stage", 0);
  segmentationNode->SetFloatProperty("midas.morph.thresholding.lower", referenceImage->GetScalarValueMin());
  segmentationNode->SetFloatProperty("midas.morph.thresholding.upper", referenceImage->GetScalarValueMin());
  segmentationNode->SetIntProperty("midas.morph.thresholding.slice", 0);
  segmentationNode->SetFloatProperty("midas.morph.erosion.threshold", referenceImage->GetScalarValueMax());
  segmentationNode->SetIntProperty("midas.morph.erosion.iterations", 0);
  segmentationNode->SetFloatProperty("midas.morph.dilation.lower", 60);
  segmentationNode->SetFloatProperty("midas.morph.dilation.upper", 160);
  segmentationNode->SetIntProperty("midas.morph.dilation.iterations", 0);
  segmentationNode->SetIntProperty("midas.morph.rethresholding.box", 0);
  segmentationNode->SetIntProperty("midas.morph.cursor.width", 1);
}

mitk::DataNode* MIDASMorphologicalSegmentorView::OnCreateNewSegmentationButtonPressed()
{
  // This creates the "final output image"... i.e. the segmentation result.
  mitk::DataNode::Pointer emptySegmentation = QmitkMIDASBaseSegmentationFunctionality::OnCreateNewSegmentationButtonPressed();
  assert(emptySegmentation);

  emptySegmentation->SetProperty(MIDASMorphologicalSegmentorView::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED.c_str(), mitk::BoolProperty::New(false));

  // Make sure we have a reference images... which should always be true at this point.
  mitk::Image* image = this->GetReferenceImageFromToolManager();
  if (image != NULL)
  {
    // Make sure we can retrieve the paintbrush tool, which can be used to create a new segmentation image.
    mitk::ToolManager* toolManager = this->GetToolManager();
    assert(toolManager);

    mitk::Tool* paintbrushTool = toolManager->GetToolById(m_PaintbrushToolId);

    if (paintbrushTool)
    {
      try
      {
        // Create that orange colour that MIDAS uses to highlight edited regions.
        mitk::ColorProperty::Pointer col = mitk::ColorProperty::New();
        col->SetColor((float)1.0, (float)(165.0/255.0), (float)0.0);

        // Create subtractions data node, and store reference to image
        mitk::DataNode::Pointer segmentationSubtractionsImageDataNode = paintbrushTool->CreateEmptySegmentationNode( image, mitk::MIDASTool::SUBTRACTIONS_IMAGE_NAME, col->GetColor());
        segmentationSubtractionsImageDataNode->SetBoolProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_NAME.c_str(), false);
        segmentationSubtractionsImageDataNode->SetBoolProperty("helper object", true);
        segmentationSubtractionsImageDataNode->SetColor(col->GetColor());
        segmentationSubtractionsImageDataNode->SetProperty("binaryimage.selectedcolor", col);
        segmentationSubtractionsImageDataNode->SetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_X.c_str(), 0);
        segmentationSubtractionsImageDataNode->SetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_Y.c_str(), 0);
        segmentationSubtractionsImageDataNode->SetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_Z.c_str(), 0);
        segmentationSubtractionsImageDataNode->SetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_X.c_str(), 0);
        segmentationSubtractionsImageDataNode->SetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_Y.c_str(), 0);
        segmentationSubtractionsImageDataNode->SetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_Z.c_str(), 0);
        segmentationSubtractionsImageDataNode->SetBoolProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_REGION_SET.c_str(), false);

        // Create additions data node, and store reference to image
        float segCol[3];
        emptySegmentation->GetColor(segCol);
        mitk::ColorProperty::Pointer segmentationColor = mitk::ColorProperty::New(segCol[0], segCol[1], segCol[2]);

        mitk::DataNode::Pointer segmentationAdditionsImageDataNode = paintbrushTool->CreateEmptySegmentationNode( image, mitk::MIDASTool::ADDITIONS_IMAGE_NAME, col->GetColor());
        segmentationAdditionsImageDataNode->SetBoolProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_NAME.c_str(), false);
        segmentationAdditionsImageDataNode->SetBoolProperty("helper object", true);
        segmentationAdditionsImageDataNode->SetBoolProperty("visible", false);
        segmentationAdditionsImageDataNode->SetColor(segCol);
        segmentationAdditionsImageDataNode->SetProperty("binaryimage.selectedcolor", segmentationColor);
        segmentationAdditionsImageDataNode->SetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_X.c_str(), 0);
        segmentationAdditionsImageDataNode->SetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_Y.c_str(), 0);
        segmentationAdditionsImageDataNode->SetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_Z.c_str(), 0);
        segmentationAdditionsImageDataNode->SetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_X.c_str(), 0);
        segmentationAdditionsImageDataNode->SetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_Y.c_str(), 0);
        segmentationAdditionsImageDataNode->SetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_Z.c_str(), 0);
        segmentationAdditionsImageDataNode->SetBoolProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_REGION_SET.c_str(), false);

        // Add the image to data storage, and specify this derived image as the one the toolManager will edit to.
        this->ApplyDisplayOptions(segmentationSubtractionsImageDataNode);
        this->ApplyDisplayOptions(segmentationAdditionsImageDataNode);
        this->GetDefaultDataStorage()->Add(segmentationSubtractionsImageDataNode, emptySegmentation); // add as a child, because the segmentation "derives" from the original
        this->GetDefaultDataStorage()->Add(segmentationAdditionsImageDataNode, emptySegmentation); // add as a child, because the segmentation "derives" from the original

        // Set working data. Compare with MIDASGeneralSegmentorView.
        mitk::ToolManager::DataVectorType workingData;
        workingData.push_back(segmentationSubtractionsImageDataNode);
        workingData.push_back(segmentationAdditionsImageDataNode);
        toolManager->SetWorkingData(workingData);

        // Set properties, and then the control values to match.
        this->SetDefaultParameterValuesFromReferenceImage();
        this->SetControlsByImageData();

        // If we are restarting a segmentation, we need to copy parameters from the previous segmentation.
        if (m_SelectedNode.IsNotNull()
            && m_SelectedImage.IsNotNull()
            && mitk::IsNodeABinaryImage(m_SelectedNode)
            && m_SelectedNode != emptySegmentation
            && CanStartSegmentationForBinaryNode(m_SelectedNode)
            )
        {
          // Copy parameters from m_SelectedNode
          int tmpInt;
          std::string tmpString;
          float tmpFloat;

          m_SelectedNode->GetIntProperty("midas.morph.stage", tmpInt);
          emptySegmentation->SetIntProperty("midas.morph.stage", tmpInt);

          m_SelectedNode->GetFloatProperty("midas.morph.thresholding.lower", tmpFloat);
          emptySegmentation->SetFloatProperty("midas.morph.thresholding.lower", tmpFloat);

          m_SelectedNode->GetFloatProperty("midas.morph.thresholding.upper", tmpFloat);
          emptySegmentation->SetFloatProperty("midas.morph.thresholding.upper", tmpFloat);

          m_SelectedNode->GetIntProperty("midas.morph.thresholding.slice", tmpInt);
          emptySegmentation->SetIntProperty("midas.morph.thresholding.slice", tmpInt);

          m_SelectedNode->GetFloatProperty("midas.morph.erosion.threshold", tmpFloat);
          emptySegmentation->SetFloatProperty("midas.morph.erosion.threshold", tmpFloat);

          m_SelectedNode->GetIntProperty("midas.morph.erosion.iterations", tmpInt);
          emptySegmentation->SetIntProperty("midas.morph.erosion.iterations", tmpInt);

          m_SelectedNode->GetFloatProperty("midas.morph.dilation.lower", tmpFloat);
          emptySegmentation->SetFloatProperty("midas.morph.dilation.lower", tmpFloat);

          m_SelectedNode->GetFloatProperty("midas.morph.dilation.upper", tmpFloat);
          emptySegmentation->SetFloatProperty("midas.morph.dilation.upper", tmpFloat);

          m_SelectedNode->GetIntProperty("midas.morph.dilation.iterations", tmpInt);
          emptySegmentation->SetIntProperty("midas.morph.dilation.iterations", tmpInt);

          m_SelectedNode->GetIntProperty("midas.morph.rethresholding.box", tmpInt);
          emptySegmentation->SetIntProperty("midas.morph.rethresholding.box", tmpInt);

          m_SelectedNode->GetIntProperty("midas.morph.cursor.width", tmpInt);
          emptySegmentation->SetIntProperty("midas.morph.cursor.width", tmpInt);

          emptySegmentation->SetBoolProperty("midas.morph.restarting", true);
        }

        // Make sure the controls match the parameters and the new segmentation is selected
        this->SetControlsByParameterValues();
        this->SelectNode(emptySegmentation);
      }
      catch (std::bad_alloc)
      {
        QMessageBox::warning(NULL,"Create new segmentation","Could not allocate memory for new segmentation");
      }
    } // end creating edit image
  }

  return emptySegmentation;
}

void MIDASMorphologicalSegmentorView::EnableSegmentationWidgets(bool b)
{
  this->m_MorphologicalControls->EnableControls(b);
}

void MIDASMorphologicalSegmentorView::NodeChanged(const mitk::DataNode* node)
{
  for (int i = 0; i < 2; i++)
  {
    mitk::DataNode::Pointer workingDataNode = this->GetToolManager()->GetWorkingData(i);
    if (workingDataNode.IsNotNull())
    {
      if (workingDataNode.GetPointer() == node)
      {
        bool isRegionSet(false);
        workingDataNode->GetBoolProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_REGION_SET.c_str(), isRegionSet);
        if (isRegionSet)
        {
          this->UpdateSegmentation();
        }
      }
    }
  }
}

void MIDASMorphologicalSegmentorView::OnThresholdingValuesChanged(double lowerThreshold, double upperThreshold, int axialSlicerNumber)
{
  mitk::DataNode* segmentationNode = this->GetSegmentationNodeUsingToolManager();
  if (segmentationNode != NULL)
  {
    segmentationNode->SetFloatProperty("midas.morph.thresholding.lower", lowerThreshold);
    segmentationNode->SetFloatProperty("midas.morph.thresholding.upper", upperThreshold);
    segmentationNode->SetIntProperty("midas.morph.thresholding.slice", axialSlicerNumber);

    this->UpdateSegmentation();
  }
}

void MIDASMorphologicalSegmentorView::OnErosionsValuesChanged(double upperThreshold, int numberOfErosions)
{
  mitk::DataNode* segmentationNode = this->GetSegmentationNodeUsingToolManager();
  if (segmentationNode != NULL)
  {
    segmentationNode->SetFloatProperty("midas.morph.erosion.threshold", upperThreshold);
    segmentationNode->SetIntProperty("midas.morph.erosion.iterations", numberOfErosions);

    this->UpdateSegmentation();
  }
}

void MIDASMorphologicalSegmentorView::OnDilationValuesChanged(double lowerPercentage, double upperPercentage, int numberOfDilations)
{
  mitk::DataNode* segmentationNode = this->GetSegmentationNodeUsingToolManager();
  if (segmentationNode != NULL)
  {
    segmentationNode->SetFloatProperty("midas.morph.dilation.lower", lowerPercentage);
    segmentationNode->SetFloatProperty("midas.morph.dilation.upper", upperPercentage);
    segmentationNode->SetIntProperty("midas.morph.dilation.iterations", numberOfDilations);

    this->UpdateSegmentation();
  }
}

void MIDASMorphologicalSegmentorView::OnRethresholdingValuesChanged(int boxSize)
{
  mitk::DataNode* segmentationNode = this->GetSegmentationNodeUsingToolManager();
  if (segmentationNode != NULL)
  {
    segmentationNode->SetIntProperty("midas.morph.rethresholing.box", boxSize);

    this->UpdateSegmentation();
  }
}

void MIDASMorphologicalSegmentorView::OnTabChanged(int i)
{

  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::DataNode* segmentationNode = this->GetSegmentationNodeUsingToolManager();
  if (segmentationNode != NULL)
  {
    if (i == 1 || i == 2) // Erosions and Dilations only
    {
      toolManager->ActivateTool(m_PaintbrushToolId);
    }
    else
    {
      toolManager->ActivateTool(-1);
    }

    segmentationNode->SetIntProperty("midas.morph.stage", i);

    this->UpdateSegmentation();
  }
}

void MIDASMorphologicalSegmentorView::OnCursorWidthChanged(int i)
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::DataNode* segmentationNode = this->GetSegmentationNodeUsingToolManager();

  if (segmentationNode != NULL)
  {
    mitk::MIDASPaintbrushTool* paintbrushTool = dynamic_cast<mitk::MIDASPaintbrushTool*>(toolManager->GetToolById(m_PaintbrushToolId));
    if (paintbrushTool != NULL)
    {
      paintbrushTool->SetCursorSize(i);

      segmentationNode->SetIntProperty("midas.morph.cursor.width", i);
    }
  }
}

void MIDASMorphologicalSegmentorView::UpdateSegmentation()
{
  mitk::DataNode::Pointer editsNode = this->GetToolManager()->GetWorkingData(0);
  if (editsNode.IsNotNull())
  {
    mitk::DataNode::Pointer additionsNode = this->GetToolManager()->GetWorkingData(1);
    if (additionsNode.IsNotNull())
    {
      mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDefaultDataStorage().GetPointer(), editsNode, true);
      if (parent.IsNotNull())
      {
        mitk::Image::Pointer outputImage    = dynamic_cast<mitk::Image*>( parent->GetData() );
        mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();  // The grey scale image
        mitk::Image::Pointer editedImage    = this->GetWorkingImageFromToolManager(0);   // Comes from tool manager, so is image of manual edits
        mitk::Image::Pointer additionsImage = this->GetWorkingImageFromToolManager(1);   // Comes from tool manager, so is image of manual additions

        if (referenceImage.IsNotNull() && editedImage.IsNotNull() && additionsImage.IsNotNull() && outputImage.IsNotNull())
        {
          bool isEditingEditingImage = false;
          editsNode->GetBoolProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_NAME.c_str(), isEditingEditingImage);

          bool isEditingAdditionsImage = false;
          additionsNode->GetBoolProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_NAME.c_str(), isEditingAdditionsImage);

          int region[6];
          for (int i = 0; i < 6; i++)
          {
            region[i] = 0;
          }

          if (isEditingEditingImage)
          {
            editsNode->GetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_X.c_str(), region[0]);
            editsNode->GetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_Y.c_str(), region[1]);
            editsNode->GetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_Z.c_str(), region[2]);
            editsNode->GetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_X.c_str(), region[3]);
            editsNode->GetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_Y.c_str(), region[4]);
            editsNode->GetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_Z.c_str(), region[5]);
          }
          else if (isEditingAdditionsImage)
          {
            additionsNode->GetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_X.c_str(), region[0]);
            additionsNode->GetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_Y.c_str(), region[1]);
            additionsNode->GetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_INDEX_Z.c_str(), region[2]);
            additionsNode->GetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_X.c_str(), region[3]);
            additionsNode->GetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_Y.c_str(), region[4]);
            additionsNode->GetIntProperty(mitk::MIDASPaintbrushTool::EDITING_PROPERTY_SIZE_Z.c_str(), region[5]);
          }

          MorphologicalSegmentorPipelineParams params;
          this->GetParameterValues(params);

          bool isRestarting(false);
          bool foundRestartingFlag = parent->GetBoolProperty("midas.morph.restarting", isRestarting);

          try
          {
            AccessFixedDimensionByItk_n(referenceImage, InvokeITKPipeline, 3, (editedImage, additionsImage, params, isEditingEditingImage, isEditingAdditionsImage, isRestarting, region, outputImage));
          }
          catch(const mitk::AccessByItkException& e)
          {
            MITK_ERROR << "Caught exception, so abandoning pipeline update:" << e.what();
          }
          catch(itk::ExceptionObject &e)
          {
            MITK_ERROR << "Caught exception, so abandoning pipeline update:" << e.what();
          }

          if (foundRestartingFlag)
          {
            parent->ReplaceProperty("midas.morph.restarting", mitk::BoolProperty::New(false));
          }

          outputImage->Modified();
          parent->Modified();
          mitk::RenderingManager::GetInstance()->RequestUpdateAll();
        }
      }

    }
  }
}

void MIDASMorphologicalSegmentorView::FinalizeSegmentation()
{
  mitk::DataNode::Pointer workingDataNode = this->GetToolManager()->GetWorkingData(0);
  if (workingDataNode.IsNotNull())
  {
    mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDefaultDataStorage().GetPointer(), workingDataNode, true);
    if (parent.IsNotNull())
    {
      mitk::Image::Pointer outputImage = mitk::Image::New();
      mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();

      try
      {
        AccessFixedDimensionByItk_n(referenceImage, FinalizeITKPipeline, 3, (outputImage));
      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Caught exception, so finalize pipeline" << e.what();
      }
      this->RemoveWorkingData();
      this->DestroyPipeline();

      parent->SetData( outputImage );
      parent->ReplaceProperty(MIDASMorphologicalSegmentorView::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED.c_str(), mitk::BoolProperty::New(true));
      this->UpdateVolumeProperty(parent);
    }
  }
}

template<typename TPixel, unsigned int VImageDimension>
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::MorphologicalSegmentorPipeline()
{
  // This is the main pipeline that will form the whole of the final output.
  m_ThresholdingFilter = ThresholdingFilterType::New();
  m_EarlyMaskFilter = MaskByRegionFilterType::New();
  m_EarlyConnectedComponentFilter = LargestConnectedComponentFilterType::New();
  m_ErosionFilter = ErosionFilterType::New();
  m_DilationFilter = DilationFilterType::New();
  m_RethresholdingFilter = RethresholdingFilterType::New();
  m_LateMaskFilter = MaskByRegionFilterType::New();
  m_LateConnectedComponentFilter = LargestConnectedComponentFilterType::New();
  m_ExcludeImageFilter = ExcludeImageFilterType::New();
  m_OrImageFilter = OrImageFilterType::New();

  // Making sure that these are only called once in constructor, to avoid unnecessary pipeline updates.
  m_ForegroundValue = 255;
  m_BackgroundValue = 0;
  m_ThresholdingFilter->SetInsideValue(m_ForegroundValue);
  m_ThresholdingFilter->SetOutsideValue(m_BackgroundValue);
  m_EarlyMaskFilter->SetOutputBackgroundValue(m_BackgroundValue);
  m_EarlyConnectedComponentFilter->SetInputBackgroundValue(m_BackgroundValue);
  m_EarlyConnectedComponentFilter->SetOutputBackgroundValue(m_BackgroundValue);
  m_EarlyConnectedComponentFilter->SetOutputForegroundValue(m_ForegroundValue);
  m_ErosionFilter->SetInValue(m_ForegroundValue);
  m_ErosionFilter->SetOutValue(m_BackgroundValue);
  m_DilationFilter->SetInValue(m_ForegroundValue);
  m_DilationFilter->SetOutValue(m_BackgroundValue);
  m_RethresholdingFilter->SetInValue(m_ForegroundValue);
  m_RethresholdingFilter->SetOutValue(m_BackgroundValue);
  m_LateMaskFilter->SetOutputBackgroundValue(m_BackgroundValue);
  m_LateConnectedComponentFilter->SetInputBackgroundValue(m_BackgroundValue);
  m_LateConnectedComponentFilter->SetOutputBackgroundValue(m_BackgroundValue);
  m_LateConnectedComponentFilter->SetOutputForegroundValue(m_ForegroundValue);
}

template<typename TPixel, unsigned int VImageDimension>
void
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::SetParam(MorphologicalSegmentorPipelineParams& p)
{
  m_Stage = p.m_Stage;

  // Note, the ITK Set/Get Macro ensures that the Modified flag only gets set if the value set is actually different.

  if (m_Stage == 0)
  {
    m_ThresholdingFilter->SetLowerThreshold((TPixel)p.m_LowerIntensityThreshold);
    m_ThresholdingFilter->SetUpperThreshold((TPixel)p.m_UpperIntensityThreshold);

    m_EarlyMaskFilter->SetInput(m_ThresholdingFilter->GetOutput());
  }
  else if (m_Stage == 1)
  {
    m_EarlyMaskFilter->SetInput(m_ThresholdingFilter->GetOutput());
    m_EarlyConnectedComponentFilter->SetInput(m_EarlyMaskFilter->GetOutput());
    m_ErosionFilter->SetBinaryImageInput(m_EarlyConnectedComponentFilter->GetOutput());
    m_ErosionFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_LateMaskFilter->SetInput(m_ErosionFilter->GetOutput());
    m_OrImageFilter->SetInput(0, m_LateMaskFilter->GetOutput());
    m_ExcludeImageFilter->SetInput(0, m_OrImageFilter->GetOutput());
    m_LateConnectedComponentFilter->SetInput(m_ExcludeImageFilter->GetOutput());

    m_ErosionFilter->SetUpperThreshold((TPixel)p.m_UpperErosionsThreshold);
    m_ErosionFilter->SetNumberOfIterations(p.m_NumberOfErosions);
  }
  else if (m_Stage == 2)
  {
    m_EarlyMaskFilter->SetInput(m_ThresholdingFilter->GetOutput());
    m_EarlyConnectedComponentFilter->SetInput(m_EarlyMaskFilter->GetOutput());
    m_ErosionFilter->SetBinaryImageInput(m_EarlyConnectedComponentFilter->GetOutput());
    m_ErosionFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_DilationFilter->SetBinaryImageInput(m_ErosionFilter->GetOutput());
    m_DilationFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_LateMaskFilter->SetInput(m_DilationFilter->GetOutput());
    m_OrImageFilter->SetInput(0, m_LateMaskFilter->GetOutput());
    m_ExcludeImageFilter->SetInput(0, m_OrImageFilter->GetOutput());
    m_LateConnectedComponentFilter->SetInput(m_ExcludeImageFilter->GetOutput());

    m_DilationFilter->SetLowerThreshold((int)(p.m_LowerPercentageThresholdForDilations));
    m_DilationFilter->SetUpperThreshold((int)(p.m_UpperPercentageThresholdForDilations));
    m_DilationFilter->SetNumberOfIterations((int)(p.m_NumberOfDilations));
  }
  else if (m_Stage == 3)
  {
    m_EarlyMaskFilter->SetInput(m_ThresholdingFilter->GetOutput());
    m_EarlyConnectedComponentFilter->SetInput(m_EarlyMaskFilter->GetOutput());
    m_ErosionFilter->SetBinaryImageInput(m_EarlyConnectedComponentFilter->GetOutput());
    m_ErosionFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_DilationFilter->SetBinaryImageInput(m_ErosionFilter->GetOutput());
    m_DilationFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_LateMaskFilter->SetInput(m_RethresholdingFilter->GetOutput());
    m_RethresholdingFilter->SetBinaryImageInput(m_DilationFilter->GetOutput());
    m_RethresholdingFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_OrImageFilter->SetInput(0, m_LateMaskFilter->GetOutput());
    m_ExcludeImageFilter->SetInput(0, m_OrImageFilter->GetOutput());
    m_LateConnectedComponentFilter->SetInput(m_ExcludeImageFilter->GetOutput());

    m_RethresholdingFilter->SetDownSamplingFactor(p.m_BoxSize);
    m_RethresholdingFilter->SetLowPercentageThreshold((int)(p.m_LowerPercentageThresholdForDilations));
    m_RethresholdingFilter->SetHighPercentageThreshold((int)(p.m_UpperPercentageThresholdForDilations));
  }
}

template<typename TPixel, unsigned int VImageDimension>
void
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::Update(bool editingImageBeingEdited, bool additionsImageBeingEdited, int *editingRegion)
{
  // Note: We try and update as small a section of the pipeline as possible.

  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::SizeType SizeType;
  typedef typename ImageType::RegionType RegionType;

  IndexType editingRegionStartIndex;
  SizeType editingRegionSize;
  RegionType editingRegionOfInterest;

  for (int i = 0; i < 3; i++)
  {
    editingRegionStartIndex[i] = editingRegion[i];
    editingRegionSize[i] = editingRegion[i + 3];
  }
  editingRegionOfInterest.SetIndex(editingRegionStartIndex);
  editingRegionOfInterest.SetSize(editingRegionSize);

  if (m_Stage == 0)
  {
    m_EarlyMaskFilter->UpdateLargestPossibleRegion();
  }
  else
  {
    if (additionsImageBeingEdited)
    {
      // Note: This little... Hacklet.. or shall we say "optimisation", basically replicates
      // the filter logic, over a tiny region of interest. I did try using filters to extract
      // a region of interest, perform the logic in another filter, and then insert the region
      // back, but it didn't work, even after sacrificing virgins to several well known deities.

      itk::ImageRegionIterator<SegmentationImageType> outputIterator(m_OrImageFilter->GetOutput(), editingRegionOfInterest);
      itk::ImageRegionConstIterator<SegmentationImageType> editedRegionIterator(m_OrImageFilter->GetInput(1), editingRegionOfInterest);
      for (outputIterator.GoToBegin(), editedRegionIterator.GoToBegin();
          !outputIterator.IsAtEnd();
          ++outputIterator, ++editedRegionIterator)
      {
        if (outputIterator.Get() > 0 || editedRegionIterator.Get() > 0)
        {
          outputIterator.Set(m_ForegroundValue);
        }
        else
        {
          outputIterator.Set(m_BackgroundValue);
        }
      }
    }
    else if (editingImageBeingEdited)
    {
      // Note: This little... Hacklet.. or shall we say "optimisation", basically replicates
      // the filter logic, over a tiny region of interest. I did try using filters to extract
      // a region of interest, perform the logic in another filter, and then insert the region
      // back, but it didn't work, even after sacrificing virgins to several well known deities.

      itk::ImageRegionIterator<SegmentationImageType> outputIterator(m_ExcludeImageFilter->GetOutput(), editingRegionOfInterest);
      itk::ImageRegionConstIterator<SegmentationImageType> editedRegionIterator(m_ExcludeImageFilter->GetInput(1), editingRegionOfInterest);
      for (outputIterator.GoToBegin(), editedRegionIterator.GoToBegin();
          !outputIterator.IsAtEnd();
          ++outputIterator, ++editedRegionIterator)
      {
        if (editedRegionIterator.Get() > 0)
        {
          outputIterator.Set(m_BackgroundValue);
        }
      }
    }
    else
    {
      // Executing the pipeline for the whole image - slow, but unavoidable.
      m_LateConnectedComponentFilter->Modified();
      m_LateConnectedComponentFilter->UpdateLargestPossibleRegion();
    }
  }
}

template<typename TPixel, unsigned int VImageDimension>
typename MorphologicalSegmentorPipeline<TPixel, VImageDimension>::SegmentationImageType::Pointer
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::GetOutput(bool editingImageBeingEdited, bool additionsImageBeingEdited)
{
  typename SegmentationImageType::Pointer result;

  if (m_Stage == 0)
  {
    result = m_EarlyMaskFilter->GetOutput();
  }
  else
  {
    if (additionsImageBeingEdited)
    {
      result = m_OrImageFilter->GetOutput();
    }
    else if (editingImageBeingEdited)
    {
      result = m_ExcludeImageFilter->GetOutput();
    }
    else
    {
      result = m_LateConnectedComponentFilter->GetOutput();
    }
  }
  return result;
}

template<typename TPixel, unsigned int VImageDimension>
void
MIDASMorphologicalSegmentorView
::InvokeITKPipeline(
    itk::Image<TPixel, VImageDimension>* itkImage,
    mitk::Image::Pointer& edits,
    mitk::Image::Pointer& additions,
    MorphologicalSegmentorPipelineParams& params,
    bool editingImageBeingUpdated,
    bool additionsImageBeingUpdated,
    bool isRestarting,
    int *editingRegion,
    mitk::Image::Pointer& output
    )
{
  typedef itk::Image<unsigned char, VImageDimension> ImageType;
  typedef mitk::ImageToItk< ImageType > ImageToItkType;

  typename ImageToItkType::Pointer editsToItk = ImageToItkType::New();
  editsToItk->SetInput(edits);
  editsToItk->Update();

  typename ImageToItkType::Pointer additionsToItk = ImageToItkType::New();
  additionsToItk->SetInput(additions);
  additionsToItk->Update();

  std::stringstream key;
  key << typeid(TPixel).name() << VImageDimension;

  MorphologicalSegmentorPipeline<TPixel, VImageDimension>* pipeline = NULL;
  MorphologicalSegmentorPipelineInterface* myPipeline = NULL;

  std::map<std::string, MorphologicalSegmentorPipelineInterface*>::iterator iter;
  iter = m_TypeToPipelineMap.find(key.str());

  if (iter == m_TypeToPipelineMap.end())
  {
    pipeline = new MorphologicalSegmentorPipeline<TPixel, VImageDimension>();
    myPipeline = pipeline;
    m_TypeToPipelineMap.insert(StringAndPipelineInterfacePair(key.str(), myPipeline));
    pipeline->m_ThresholdingFilter->SetInput(itkImage);
    pipeline->m_ExcludeImageFilter->SetInput(1, editsToItk->GetOutput());
    pipeline->m_OrImageFilter->SetInput(1, additionsToItk->GetOutput());
  }
  else
  {
    myPipeline = iter->second;
    pipeline = static_cast<MorphologicalSegmentorPipeline<TPixel, VImageDimension>*>(myPipeline);
  }

  // Set most of the parameters on the pipeline.
  pipeline->SetParam(params);

  // Start Trac 998, setting region of interest, on both Mask filters, to produce Axial-Cut-off effect.

  typename ImageType::RegionType regionOfInterest;
  typename ImageType::SizeType   regionOfInterestSize;
  typename ImageType::IndexType  regionOfInterestIndex;

  // 1. Set region to full size of input image
  regionOfInterestSize = itkImage->GetLargestPossibleRegion().GetSize();
  regionOfInterestIndex = itkImage->GetLargestPossibleRegion().GetIndex();

  // 2. Get string describing orientation.
  typename itk::SpatialOrientationAdapter adaptor;
  typename itk::SpatialOrientation::ValidCoordinateOrientationFlags orientation;
  orientation = adaptor.FromDirectionCosines(itkImage->GetDirection());
  std::string orientationString = itk::ConvertSpatialOrientationToString(orientation);

  // 3. Get Axis that represents superior/inferior
  int axialAxis = QmitkMIDASBaseSegmentationFunctionality::GetAxialAxis();
  if (axialAxis != -1)
  {
    // 4. Calculate size of region of interest in that axis
    regionOfInterestSize[axialAxis] = regionOfInterestSize[axialAxis] - params.m_AxialCutoffSlice - 1;
    if (orientationString[axialAxis] == 'I')
    {
      regionOfInterestIndex[axialAxis] = regionOfInterestIndex[axialAxis] + params.m_AxialCutoffSlice;
    }

    // 5. Set region on both filters
    regionOfInterest.SetSize(regionOfInterestSize);
    regionOfInterest.SetIndex(regionOfInterestIndex);
    pipeline->m_LateMaskFilter->SetRegion(regionOfInterest);
  }

  // End Trac 998, setting region of interest, on both Mask filters

  // Start Trac 1131, calculate a rough size to help LargestConnectedComponents allocate memory.

  unsigned long int expectedSize = 1;
  for (unsigned int i = 0; i < VImageDimension; i++)
  {
    expectedSize *= regionOfInterestSize[i];
  }
  expectedSize /= 8;

  // However, make sure we only update the minimum amount possible.
  if (params.m_Stage == 0)
  {
    pipeline->m_EarlyMaskFilter->SetRegion(regionOfInterest);
    pipeline->m_EarlyConnectedComponentFilter->SetCapacity(expectedSize);
  }
  else
  {
    pipeline->m_LateMaskFilter->SetRegion(regionOfInterest);
    pipeline->m_LateConnectedComponentFilter->SetCapacity(expectedSize);
  }

  // End Trac 1131.

  // Do the update.
  if (isRestarting)
  {
    for (int i = 0; i <= params.m_Stage; i++)
    {
      params.m_Stage = i;
      pipeline->SetParam(params);
      pipeline->Update(editingImageBeingUpdated, additionsImageBeingUpdated, editingRegion);
    }
  }
  else
  {
    pipeline->Update(editingImageBeingUpdated, additionsImageBeingUpdated, editingRegion);
  }

  // Get hold of the output, and make sure we don't re-allocate memory.
  output->InitializeByItk< ImageType >(pipeline->GetOutput(editingImageBeingUpdated, additionsImageBeingUpdated).GetPointer());
  output->SetImportChannel(pipeline->GetOutput(editingImageBeingUpdated, additionsImageBeingUpdated)->GetBufferPointer(), 0, mitk::Image::ReferenceMemory);

  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

template<typename TPixel, unsigned int VImageDimension>
void
MIDASMorphologicalSegmentorView
::FinalizeITKPipeline(
    itk::Image<TPixel, VImageDimension>* itkImage,
    mitk::Image::Pointer& output
    )
{
  typedef itk::Image<unsigned char, VImageDimension> ImageType;

  std::stringstream key;
  key << typeid(TPixel).name() << VImageDimension;

  MorphologicalSegmentorPipeline<TPixel, VImageDimension>* pipeline = NULL;
  MorphologicalSegmentorPipelineInterface* myPipeline = NULL;

  std::map<std::string, MorphologicalSegmentorPipelineInterface*>::iterator iter;
  iter = m_TypeToPipelineMap.find(key.str());

  // By the time this method is called, the pipeline MUST exist.
  myPipeline = iter->second;
  pipeline = static_cast<MorphologicalSegmentorPipeline<TPixel, VImageDimension>*>(myPipeline);

  // This deliberately re-allocates the memory
  mitk::CastToMitkImage(pipeline->GetOutput(false, false), output);
}

void MIDASMorphologicalSegmentorView::ClearWorkingData()
{
  mitk::Image::Pointer editsImage = this->GetWorkingImageFromToolManager(0);
  mitk::Image::Pointer additionsImage = this->GetWorkingImageFromToolManager(1);
  mitk::DataNode::Pointer editsNode = this->GetToolManager()->GetWorkingData(0);
  mitk::DataNode::Pointer additionsNode = this->GetToolManager()->GetWorkingData(1);

  if (editsImage.IsNotNull() && additionsImage.IsNotNull() && editsNode.IsNotNull() && additionsNode.IsNotNull())
  {
    try
    {
      AccessFixedDimensionByItk(editsImage, ClearITKImage, 3);
      AccessFixedDimensionByItk(additionsImage, ClearITKImage, 3);

      editsImage->Modified();
      additionsImage->Modified();
      editsNode->Modified();
      additionsNode->Modified();
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so abandoning clearing the segmentation image:" << e.what();
    }
  }
}

template<typename TPixel, unsigned int VImageDimension>
void
MIDASMorphologicalSegmentorView
::ClearITKImage(itk::Image<TPixel, VImageDimension>* itkImage)
{
  itkImage->FillBuffer(0);
}

void MIDASMorphologicalSegmentorView::SetControlsByImageData()
{
  mitk::Image* image = this->GetReferenceImageFromToolManager();

  if (image != NULL)
  {
    int axialAxis = QmitkMIDASBaseSegmentationFunctionality::GetAxialAxis();
    int numberOfAxialSlices = image->GetDimension(axialAxis);

    m_MorphologicalControls->SetControlsByImageData(
        image->GetScalarValueMin(),
        image->GetScalarValueMax(),
        numberOfAxialSlices);
  }
}

void MIDASMorphologicalSegmentorView::GetParameterValues(MorphologicalSegmentorPipelineParams& params)
{
  mitk::DataNode::Pointer workingDataNode = this->GetToolManager()->GetWorkingData(0);
  if (workingDataNode.IsNotNull())
  {
    mitk::DataNode::Pointer segmentationDataNode = mitk::FindFirstParentImage(this->GetDefaultDataStorage().GetPointer(), workingDataNode, true);
    if (segmentationDataNode.IsNotNull())
    {
      segmentationDataNode->GetIntProperty("midas.morph.stage", params.m_Stage);
      segmentationDataNode->GetFloatProperty("midas.morph.thresholding.lower", params.m_LowerIntensityThreshold);
      segmentationDataNode->GetFloatProperty("midas.morph.thresholding.upper", params.m_UpperIntensityThreshold);
      segmentationDataNode->GetIntProperty("midas.morph.thresholding.slice", params.m_AxialCutoffSlice);
      segmentationDataNode->GetFloatProperty("midas.morph.erosion.threshold", params.m_UpperErosionsThreshold);
      segmentationDataNode->GetIntProperty("midas.morph.erosion.iterations", params.m_NumberOfErosions);
      segmentationDataNode->GetFloatProperty("midas.morph.dilation.lower", params.m_LowerPercentageThresholdForDilations);
      segmentationDataNode->GetFloatProperty("midas.morph.dilation.upper", params.m_UpperPercentageThresholdForDilations);
      segmentationDataNode->GetIntProperty("midas.morph.dilation.iterations", params.m_NumberOfDilations);
      segmentationDataNode->GetIntProperty("midas.morph.rethresholding.box", params.m_BoxSize);
      segmentationDataNode->GetIntProperty("midas.morph.cursor.width", params.m_CursorWidth);
    }
  }
}

void MIDASMorphologicalSegmentorView::SetControlsByParameterValues()
{
  MorphologicalSegmentorPipelineParams params;
  this->GetParameterValues(params);

  this->SetControlsByImageData();
  this->m_MorphologicalControls->SetControlsByParameterValues(params);
}

void MIDASMorphologicalSegmentorView::RemoveWorkingData()
{
  mitk::DataNode::Pointer editsNode = this->GetToolManager()->GetWorkingData(0);
  assert(editsNode);

  mitk::DataNode::Pointer additionsNode = this->GetToolManager()->GetWorkingData(1);
  assert(additionsNode);

  this->GetDefaultDataStorage()->Remove(editsNode);
  this->GetDefaultDataStorage()->Remove(additionsNode);

  mitk::ToolManager* toolManager = this->GetToolManager();
  mitk::ToolManager::DataVectorType emptyWorkingDataArray;
  toolManager->SetWorkingData(emptyWorkingDataArray);
  toolManager->ActivateTool(-1);
}

void MIDASMorphologicalSegmentorView::DestroyPipeline()
{
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    try
    {
      AccessFixedDimensionByItk(referenceImage, DestroyITKPipeline, 3);
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so abandoning clearing the segmentation image:" << e.what();
    }
  }
}

template<typename TPixel, unsigned int VImageDimension>
void
MIDASMorphologicalSegmentorView
::DestroyITKPipeline(itk::Image<TPixel, VImageDimension>* itkImage)
{

  std::stringstream key;
  key << typeid(TPixel).name() << VImageDimension;

  std::map<std::string, MorphologicalSegmentorPipelineInterface*>::iterator iter;
  iter = m_TypeToPipelineMap.find(key.str());

  MorphologicalSegmentorPipeline<TPixel, VImageDimension> *pipeline = dynamic_cast<MorphologicalSegmentorPipeline<TPixel, VImageDimension>*>(iter->second);
  if (pipeline != NULL)
  {
    delete pipeline;
  }
  else
  {
    MITK_ERROR << "MIDASMorphologicalSegmentorView::DestroyITKPipeline(..), failed to delete pipeline" << std::endl;
  }
  m_TypeToPipelineMap.clear();
}

void MIDASMorphologicalSegmentorView::OnClearButtonClicked()
{
  this->ClearWorkingData();
  this->SetDefaultParameterValuesFromReferenceImage();
  this->SetControlsByImageData();
  this->SetControlsByParameterValues();
  this->UpdateSegmentation();
  this->FireNodeSelected(this->GetSegmentationNodeUsingToolManager());
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

void MIDASMorphologicalSegmentorView::OnOKButtonClicked()
{
  this->FinalizeSegmentation();
  this->SetReferenceImageSelected();
  this->EnableSegmentationWidgets(false);
  m_MorphologicalControls->m_TabWidget->blockSignals(true);
  m_MorphologicalControls->m_TabWidget->setCurrentIndex(0);
  m_MorphologicalControls->m_TabWidget->blockSignals(false);
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

void MIDASMorphologicalSegmentorView::OnCancelButtonClicked()
{
  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNodeUsingToolManager();
  assert(segmentationNode);

  this->ClearWorkingData();
  this->DestroyPipeline();
  this->RemoveWorkingData();
  this->GetDefaultDataStorage()->Remove(segmentationNode);
  this->SetReferenceImageSelected();
  this->EnableSegmentationWidgets(false);
  m_MorphologicalControls->m_TabWidget->blockSignals(true);
  m_MorphologicalControls->m_TabWidget->setCurrentIndex(0);
  m_MorphologicalControls->m_TabWidget->blockSignals(false);
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

