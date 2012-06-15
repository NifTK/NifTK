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

#include "berryIWorkbenchPage.h"
#include "mitkImageAccessByItk.h"
#include "mitkITKImageImport.h"
#include "mitkImageCast.h"
#include "mitkImage.h"
#include "mitkImageStatisticsHolder.h"
#include "mitkMIDASTool.h"
#include "mitkMIDASPaintbrushTool.h"
#include "mitkColorProperty.h"
#include "mitkDataStorageUtils.h"
#include "mitkITKRegionParametersDataNodeProperty.h"
#include "mitkSegmentationObjectFactory.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkConversionUtils.h"
#include "QmitkMIDASMultiViewWidget.h"

const std::string MIDASMorphologicalSegmentorView::VIEW_ID = "uk.ac.ucl.cmic.midasmorphologicalsegmentor";
const std::string MIDASMorphologicalSegmentorView::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED = "midas.morph.finished";

MIDASMorphologicalSegmentorView::MIDASMorphologicalSegmentorView()
: QmitkMIDASBaseSegmentationFunctionality()
, m_MorphologicalControls(NULL)
, m_Layout(NULL)
, m_ContainerForSelectorWidget(NULL)
, m_ContainerForToolWidget(NULL)
, m_ContainerForControlsWidget(NULL)
, m_PaintbrushToolId(-1)
{
  RegisterSegmentationObjectFactory();
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

void MIDASMorphologicalSegmentorView::CreateQtPartControl(QWidget *parent)
{
  m_Parent = parent;

  if (!m_MorphologicalControls)
  {
    m_Layout = new QGridLayout(parent);

    m_ContainerForSelectorWidget = new QWidget(parent);
    m_ContainerForToolWidget = new QWidget(parent);
    m_ContainerForControlsWidget = new QWidget(parent);

    m_MorphologicalControls = new MIDASMorphologicalSegmentorViewControlsImpl();
    m_MorphologicalControls->setupUi(m_ContainerForControlsWidget);
    m_MorphologicalControls->m_TabWidget->setCurrentIndex(0);

    QmitkMIDASBaseSegmentationFunctionality::CreateQtPartControl(m_ContainerForSelectorWidget, m_ContainerForToolWidget);

    m_Layout->addWidget(m_ContainerForSelectorWidget, 0, 0);
    m_Layout->addWidget(m_ContainerForToolWidget,     1, 0);
    m_Layout->addWidget(m_ContainerForControlsWidget, 2, 0);

    m_ToolSelector->m_ManualToolSelectionBox->SetDisplayedToolGroups("Paintbrush");

    mitk::ToolManager* toolManager = this->GetToolManager();
    m_PaintbrushToolId = toolManager->GetToolIdByToolType<mitk::MIDASPaintbrushTool>();

    this->CreateConnections();
  }
}

void MIDASMorphologicalSegmentorView::CreateConnections()
{
  QmitkMIDASBaseSegmentationFunctionality::CreateConnections();

  if (m_MorphologicalControls)
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

void MIDASMorphologicalSegmentorView::SetFocus()
{
}

void MIDASMorphologicalSegmentorView::ClosePart()
{
  mitk::DataNode* segmentationNode = this->GetSegmentationNodeUsingToolManager();
  if (segmentationNode != NULL)
  {
    this->OnCancelButtonClicked();
  }
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

    mitk::DataNode::Pointer parent = FindFirstParentImage(this->GetDataStorage(), node, false);

    if (parent.IsNotNull())
    {
      // Should also have two children called SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME
      mitk::DataStorage::SetOfObjects::Pointer children = FindDerivedImages(this->GetDataStorage(), node, true);
      if (children->size() == 2)
      {
        std::string name1;
        (*children)[0]->GetStringProperty("name", name1);
        std::string name2;
        (*children)[1]->GetStringProperty("name", name2);

        if ((name1 == mitk::MIDASTool::MORPH_EDITS_SUBTRACTIONS || name1 == mitk::MIDASTool::MORPH_EDITS_ADDITIONS)
            && (name2 == mitk::MIDASTool::MORPH_EDITS_SUBTRACTIONS || name2 == mitk::MIDASTool::MORPH_EDITS_ADDITIONS)
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
    mitk::DataNode::Pointer parent = FindFirstParentImage(this->GetDataStorage(), node, true);

    if (parent.IsNotNull())
    {
      std::string name;
      if (node->GetStringProperty("name", name))
      {
        if (name == mitk::MIDASTool::MORPH_EDITS_SUBTRACTIONS || name == mitk::MIDASTool::MORPH_EDITS_ADDITIONS)
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

  mitk::DataStorage::SetOfObjects::Pointer children = FindDerivedImages(this->GetDataStorage(), node, true );

  for (unsigned int i = 0; i < children->size(); i++)
  {
    std::string name;
    if ((*children)[i]->GetStringProperty("name", name))
    {
      if (name == mitk::MIDASTool::MORPH_EDITS_ADDITIONS)
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
      if (name == mitk::MIDASTool::MORPH_EDITS_SUBTRACTIONS)
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
    mitk::DataNode::Pointer parent = FindFirstParentImage(this->GetDataStorage(), node, true);
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
    result = FindFirstParentImage(this->GetDataStorage(), workingNodesFromToolManager[0], true);
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

void MIDASMorphologicalSegmentorView::OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes)
{
  QmitkMIDASBaseSegmentationFunctionality::OnSelectionChanged(part, nodes);

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


void MIDASMorphologicalSegmentorView::SetDefaultParameterValuesFromReferenceImage()
{
  mitk::Image* referenceImage = this->GetReferenceImage();
  assert(referenceImage);

  mitk::DataNode* segmentationNode = this->GetSegmentationNodeUsingToolManager();
  assert(segmentationNode);

  segmentationNode->SetIntProperty("midas.morph.stage", 0);
  segmentationNode->SetFloatProperty("midas.morph.thresholding.lower", referenceImage->GetStatistics()->GetScalarValueMin());
  segmentationNode->SetFloatProperty("midas.morph.thresholding.upper", referenceImage->GetStatistics()->GetScalarValueMin());
  segmentationNode->SetIntProperty("midas.morph.thresholding.slice", 0);
  segmentationNode->SetFloatProperty("midas.morph.erosion.threshold", referenceImage->GetStatistics()->GetScalarValueMax());
  segmentationNode->SetIntProperty("midas.morph.erosion.iterations", 0);
  segmentationNode->SetFloatProperty("midas.morph.dilation.lower", 60);
  segmentationNode->SetFloatProperty("midas.morph.dilation.upper", 160);
  segmentationNode->SetIntProperty("midas.morph.dilation.iterations", 0);
  segmentationNode->SetIntProperty("midas.morph.rethresholding.box", 0);
}

mitk::DataNode* MIDASMorphologicalSegmentorView::OnCreateNewSegmentationButtonPressed()
{
  // Create the new segmentation, either using a previously selected one, or create a new volume.
  mitk::DataNode::Pointer newSegmentation = NULL;
  bool isRestarting = false;

  // Make sure we have a reference images... which should always be true at this point.
  mitk::Image* image = this->GetReferenceImageFromToolManager();
  if (image != NULL)
  {

    // Make sure we can retrieve the paintbrush tool, which can be used to create a new segmentation image.
    mitk::ToolManager* toolManager = this->GetToolManager();
    assert(toolManager);

    mitk::Tool* paintbrushTool = toolManager->GetToolById(m_PaintbrushToolId);
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

      // The above method returns NULL if the use exited the colour selection dialog box.
      if (newSegmentation.IsNull())
      {
        return NULL;
      }
    }

    // Mark the newSegmentation as "unfinished".
    newSegmentation->SetProperty(MIDASMorphologicalSegmentorView::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED.c_str(), mitk::BoolProperty::New(false));

    try
    {
      // Create that orange colour that MIDAS uses to highlight edited regions.
      mitk::ColorProperty::Pointer col = mitk::ColorProperty::New();
      col->SetColor((float)1.0, (float)(165.0/255.0), (float)0.0);

      // Create subtractions data node, and store reference to image
      mitk::DataNode::Pointer segmentationSubtractionsImageDataNode = paintbrushTool->CreateEmptySegmentationNode( image, mitk::MIDASTool::MORPH_EDITS_SUBTRACTIONS, col->GetColor());
      segmentationSubtractionsImageDataNode->SetBoolProperty("helper object", true);
      segmentationSubtractionsImageDataNode->SetColor(col->GetColor());
      segmentationSubtractionsImageDataNode->SetProperty("binaryimage.selectedcolor", col);

      // Create additions data node, and store reference to image
      float segCol[3];
      newSegmentation->GetColor(segCol);
      mitk::ColorProperty::Pointer segmentationColor = mitk::ColorProperty::New(segCol[0], segCol[1], segCol[2]);

      mitk::DataNode::Pointer segmentationAdditionsImageDataNode = paintbrushTool->CreateEmptySegmentationNode( image, mitk::MIDASTool::MORPH_EDITS_ADDITIONS, col->GetColor());
      segmentationAdditionsImageDataNode->SetBoolProperty("helper object", true);
      segmentationAdditionsImageDataNode->SetBoolProperty("visible", false);
      segmentationAdditionsImageDataNode->SetColor(segCol);
      segmentationAdditionsImageDataNode->SetProperty("binaryimage.selectedcolor", segmentationColor);

      // Add the image to data storage, and specify this derived image as the one the toolManager will edit to.
      this->ApplyDisplayOptions(segmentationSubtractionsImageDataNode);
      this->ApplyDisplayOptions(segmentationAdditionsImageDataNode);
      this->GetDataStorage()->Add(segmentationSubtractionsImageDataNode, newSegmentation); // add as a child, because the segmentation "derives" from the original
      this->GetDataStorage()->Add(segmentationAdditionsImageDataNode, newSegmentation); // add as a child, because the segmentation "derives" from the original

      // Set working data. Compare with MIDASGeneralSegmentorView.
      // Note the order:
      //
      // 1. The First image is the "Additions" image, that we can manually add data/voxels to.
      // 2. The Second image is the "Subtractions" image, that is used for connection breaker.
      //
      // This must match the order in:
      //
      // 1. UpdateSegmentation
      // 2. mitkMIDASPaintbrushTool.

      mitk::ToolManager::DataVectorType workingData;
      workingData.push_back(segmentationAdditionsImageDataNode);
      workingData.push_back(segmentationSubtractionsImageDataNode);
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
      this->SelectNode(newSegmentation);
    }
    catch (std::bad_alloc)
    {
      QMessageBox::warning(NULL,"Create new segmentation","Could not allocate memory for new segmentation");
    }

  } // end if we have a reference image

  // And... relax.
  return newSegmentation;
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
        mitk::ITKRegionParametersDataNodeProperty::Pointer prop = static_cast<mitk::ITKRegionParametersDataNodeProperty*>(workingDataNode->GetProperty(mitk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str()));
        if (prop.IsNotNull() && prop->HasVolume())
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
  mitk::DataNode* segmentationNode = this->GetSegmentationNodeUsingToolManager();
  if (segmentationNode != NULL)
  {
    if (i == 1 || i == 2)
    {
      this->m_ToolSelector->SetEnabled(true);
    }
    else
    {
      this->m_ToolSelector->SetEnabled(false);
      this->OnToolSelected(-1); // make sure we de-activate tools.
    }

    segmentationNode->SetIntProperty("midas.morph.stage", i);

    this->UpdateSegmentation();
  }
}

void MIDASMorphologicalSegmentorView::UpdateSegmentation()
{
  mitk::DataNode::Pointer additionsNode = this->GetToolManager()->GetWorkingData(0);
  if (additionsNode.IsNotNull())
  {
    mitk::DataNode::Pointer editsNode = this->GetToolManager()->GetWorkingData(1);
    if (editsNode.IsNotNull())
    {
      mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDataStorage().GetPointer(), editsNode, true);
      if (parent.IsNotNull())
      {
        mitk::Image::Pointer outputImage    = dynamic_cast<mitk::Image*>( parent->GetData() );
        mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();  // The grey scale image
        mitk::Image::Pointer additionsImage = this->GetWorkingImageFromToolManager(0);   // Comes from tool manager, so is image of manual additions
        mitk::Image::Pointer editedImage    = this->GetWorkingImageFromToolManager(1);   // Comes from tool manager, so is image of manual edits

        if (referenceImage.IsNotNull() && editedImage.IsNotNull() && additionsImage.IsNotNull() && outputImage.IsNotNull())
        {

          MorphologicalSegmentorPipelineParams params;
          this->GetParameterValues(params);

          std::vector<int> region;
          region.resize(6);

          bool isRestarting(false);
          bool foundRestartingFlag = parent->GetBoolProperty("midas.morph.restarting", isRestarting);

          bool isEditingEditingImage = false;
          mitk::ITKRegionParametersDataNodeProperty::Pointer editingImageRegionProp = static_cast<mitk::ITKRegionParametersDataNodeProperty*>(editsNode->GetProperty(mitk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str()));
          if (editingImageRegionProp.IsNotNull())
          {
            isEditingEditingImage = editingImageRegionProp->IsValid();
            if (isEditingEditingImage)
            {
              region = editingImageRegionProp->GetITKRegionParameters();
            }
          }

          bool isEditingAdditionsImage = false;
          mitk::ITKRegionParametersDataNodeProperty::Pointer additionsImageRegionProp = static_cast<mitk::ITKRegionParametersDataNodeProperty*>(additionsNode->GetProperty(mitk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str()));
          if (additionsImageRegionProp.IsNotNull())
          {
            isEditingAdditionsImage = additionsImageRegionProp->IsValid();
            if (isEditingAdditionsImage)
            {
              region = additionsImageRegionProp->GetITKRegionParameters();
            }
          }

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

          QmitkAbstractView::RequestRenderWindowUpdate();
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
    mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDataStorage().GetPointer(), workingDataNode, true);
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
    std::vector<int>& editingRegion,
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
    pipeline->m_ErosionMaskFilter->SetInput(2, editsToItk->GetOutput());
    pipeline->m_ErosionMaskFilter->SetInput(1, additionsToItk->GetOutput());
    pipeline->m_DilationMaskFilter->SetInput(2, editsToItk->GetOutput());
    pipeline->m_DilationMaskFilter->SetInput(1, additionsToItk->GetOutput());
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
  int axialAxis = QmitkMIDASBaseSegmentationFunctionality::GetReferenceImageAxialAxis();
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
    pipeline->m_ErosionMaskFilter->SetRegion(regionOfInterest);
    pipeline->m_DilationMaskFilter->SetRegion(regionOfInterest);
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
    pipeline->m_ErosionMaskFilter->SetRegion(regionOfInterest);
    pipeline->m_DilationMaskFilter->SetRegion(regionOfInterest);
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

  QmitkAbstractView::RequestRenderWindowUpdate();
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
  mitk::Image::Pointer additionsImage = this->GetWorkingImageFromToolManager(0);
  mitk::Image::Pointer editsImage = this->GetWorkingImageFromToolManager(1);
  mitk::DataNode::Pointer additionsNode = this->GetToolManager()->GetWorkingData(0);
  mitk::DataNode::Pointer editsNode = this->GetToolManager()->GetWorkingData(1);

  if (editsImage.IsNotNull() && additionsImage.IsNotNull() && editsNode.IsNotNull() && additionsNode.IsNotNull())
  {
    try
    {
      AccessFixedDimensionByItk(additionsImage, ClearITKImage, 3);
      AccessFixedDimensionByItk(editsImage, ClearITKImage, 3);

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
    int axialAxis = QmitkMIDASBaseSegmentationFunctionality::GetReferenceImageAxialAxis();
    int numberOfAxialSlices = image->GetDimension(axialAxis);

    m_MorphologicalControls->SetControlsByImageData(
        image->GetStatistics()->GetScalarValueMin(),
        image->GetStatistics()->GetScalarValueMax(),
        numberOfAxialSlices);
  }
}

void MIDASMorphologicalSegmentorView::GetParameterValues(MorphologicalSegmentorPipelineParams& params)
{
  mitk::DataNode::Pointer workingDataNode = this->GetToolManager()->GetWorkingData(0);
  if (workingDataNode.IsNotNull())
  {
    mitk::DataNode::Pointer segmentationDataNode = mitk::FindFirstParentImage(this->GetDataStorage().GetPointer(), workingDataNode, true);
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
  mitk::DataNode::Pointer additionsNode = this->GetToolManager()->GetWorkingData(0);
  assert(additionsNode);

  mitk::DataNode::Pointer editsNode = this->GetToolManager()->GetWorkingData(1);
  assert(editsNode);

  this->GetDataStorage()->Remove(additionsNode);
  this->GetDataStorage()->Remove(editsNode);

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
  this->OnToolSelected(-1);
  QmitkAbstractView::RequestRenderWindowUpdate();
}

void MIDASMorphologicalSegmentorView::OnOKButtonClicked()
{
  this->FinalizeSegmentation();
  this->SetReferenceImageSelected();
  this->OnToolSelected(-1);
  this->EnableSegmentationWidgets(false);
  m_MorphologicalControls->m_TabWidget->blockSignals(true);
  m_MorphologicalControls->m_TabWidget->setCurrentIndex(0);
  m_MorphologicalControls->m_TabWidget->blockSignals(false);
  QmitkAbstractView::RequestRenderWindowUpdate();
}

void MIDASMorphologicalSegmentorView::OnCancelButtonClicked()
{
  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNodeUsingToolManager();
  assert(segmentationNode);

  this->ClearWorkingData();
  this->DestroyPipeline();
  this->RemoveWorkingData();
  this->GetDataStorage()->Remove(segmentationNode);
  this->OnToolSelected(-1);
  this->EnableSegmentationWidgets(false);
  m_MorphologicalControls->m_TabWidget->blockSignals(true);
  m_MorphologicalControls->m_TabWidget->setCurrentIndex(0);
  m_MorphologicalControls->m_TabWidget->blockSignals(false);
  QmitkAbstractView::RequestRenderWindowUpdate();
}

