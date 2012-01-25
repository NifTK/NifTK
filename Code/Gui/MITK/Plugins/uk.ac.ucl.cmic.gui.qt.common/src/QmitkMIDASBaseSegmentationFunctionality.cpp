/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-20 13:19:18 +0000 (Sun, 20 Nov 2011) $
 Revision          : $Revision: 7816 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkMIDASBaseSegmentationFunctionality.h"

#include <QMessageBox>

#include "mitkImageAccessByItk.h"
#include "mitkDataNodeObject.h"
#include "mitkNodePredicateDataType.h"
#include "mitkNodePredicateProperty.h"
#include "mitkNodePredicateAnd.h"
#include "mitkNodePredicateNot.h"
#include "mitkProperties.h"
#include "mitkColorProperty.h"
#include "mitkRenderingManager.h"
#include "mitkSegTool2D.h"
#include "mitkVtkResliceInterpolationProperty.h"
#include "mitkPointSet.h"
#include "mitkToolManager.h"
#include "mitkMIDASTool.h"
#include "mitkGlobalInteraction.h"
#include "mitkDataStorageUtils.h"
#include "mitkColorProperty.h"
#include "mitkProperties.h"
#include "mitkMIDASTool.h"
#include "mitkMIDASDrawTool.h"
#include "mitkMIDASPolyTool.h"
#include "mitkMIDASSeedTool.h"
#include "QmitkMIDASNewSegmentationDialog.h"
#include "QmitkRenderWindow.h"

#include "NifTKConfigure.h"
#include "itkConversionUtils.h"

QmitkMIDASBaseSegmentationFunctionality::QmitkMIDASBaseSegmentationFunctionality()
: m_ImageAndSegmentationSelector(NULL)
{
  m_SelectedNode = NULL;
}

QmitkMIDASBaseSegmentationFunctionality::~QmitkMIDASBaseSegmentationFunctionality()
{
  if (m_ImageAndSegmentationSelector != NULL)
  {
    delete m_ImageAndSegmentationSelector;
  }
}

QmitkMIDASBaseSegmentationFunctionality::QmitkMIDASBaseSegmentationFunctionality(
    const QmitkMIDASBaseSegmentationFunctionality& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

void QmitkMIDASBaseSegmentationFunctionality::NewNodesGenerated()
{
  std::cerr << "Matt, not finished yet, QmitkMIDASBaseSegmentationFunctionality::NewNodesGenerated" << std::endl;
}

void QmitkMIDASBaseSegmentationFunctionality::NewNodeObjectsGenerated(mitk::ToolManager::DataVectorType* nodes)
{
  std::cerr << "Matt, not finished yet, QmitkMIDASBaseSegmentationFunctionality::NewNodeObjectsGenerated" << std::endl;
}

void QmitkMIDASBaseSegmentationFunctionality::FireNodeSelected( mitk::DataNode* node )
{
  std::vector<mitk::DataNode*> nodes;
  nodes.push_back(node);
  this->FireNodesSelected(nodes);
}

void QmitkMIDASBaseSegmentationFunctionality::FireNodesSelected( std::vector<mitk::DataNode*> nodes )
{
  std::cerr << "Matt, not finished yet, QmitkMIDASBaseSegmentationFunctionality::FireNodesSelected " << std::endl;
}

void QmitkMIDASBaseSegmentationFunctionality::SelectNode(const mitk::DataNode::Pointer node)
{
  assert(node);
  this->FireNodeSelected(node);
  this->OnSelectionChanged(node);
}

void QmitkMIDASBaseSegmentationFunctionality::OnSelectionChanged(mitk::DataNode* node)
{
  std::cerr << "Matt, QmitkMIDASBaseSegmentationFunctionality::OnSelectionChanged" << std::endl;

  std::vector<mitk::DataNode*> nodes;
  nodes.push_back( node );
  this->OnSelectionChanged( nodes );
}

void QmitkMIDASBaseSegmentationFunctionality::OnSelectionChanged(std::vector<mitk::DataNode*> nodes)
{
  std::cerr << "Matt, QmitkMIDASBaseSegmentationFunctionality::OnSelectionChanged (nodes)" << std::endl;

  // If the plugin is not visible, then we have nothing to do.
  if (!m_Parent || !m_Parent->isVisible()) return;

  // By default, assume we are not going to enable the controls.
  bool valid = false;

  // This plugin only works if you single select, anything else is invalid (for now).
  if (nodes.size() == 1)
  {

    m_SelectedNode = nodes[0];
    m_SelectedImage = dynamic_cast<mitk::Image*>(m_SelectedNode->GetData());

    // MAJOR ASSUMPTION: To get a segmentation plugin (i.e. all derived classes) to work, you select the segmentation node.
    // From this segmentation node, you can work out the reference data (always the parent).
    // In addition, you can work out any intermediate working images (either that image, or children).
    // MAJOR ASSUMPTION: Intermediate working images will be hidden, and hence not clickable.

    mitk::DataNode::Pointer node = nodes[0];
    mitk::DataNode::Pointer referenceData = 0;
    mitk::DataNode::Pointer segmentedData = 0;
    mitk::ToolManager::DataVectorType workingDataNodes;

    // Rely on subclasses deciding if the node is something we are interested in.
    if (this->IsNodeAReferenceImage(node))
    {
      referenceData = node;
    }

    // A segmentation image, is the final output, the one being segmented.
    if (this->IsNodeASegmentationImage(node))
    {
      segmentedData = node;
    }

    bool test1 = mitk::IsNodeABinaryImage(node);
    bool test2 = this->CanStartSegmentationForBinaryNode(node);
    bool test3 = !this->IsNodeASegmentationImage(node);

    if (test1 && test2 && test3)
    {
      segmentedData = node;
    }

    if (segmentedData.IsNotNull())
    {

      referenceData = this->GetReferenceNodeFromSegmentationNode(segmentedData);

      if (this->IsNodeASegmentationImage(node))
      {
        workingDataNodes = this->GetWorkingNodesFromSegmentationNode(segmentedData);
        valid = true;
      }
    }

    // If we have worked out the reference data, then set the combo box.
    if (referenceData.IsNotNull())
    {
      this->m_ImageAndSegmentationSelector->m_ImageToSegmentComboBox->blockSignals(true);
      this->m_ImageAndSegmentationSelector->m_ImageToSegmentComboBox->setCurrentIndex(m_ImageAndSegmentationSelector->m_ImageToSegmentComboBox->Find(referenceData));
      this->m_ImageAndSegmentationSelector->m_ImageToSegmentComboBox->blockSignals(false);
    }

    // Tell the tool manager the images for reference and working purposes.
    this->SetToolManagerSelection(referenceData, workingDataNodes);

  }

  // Adjust widgets according to whether we have a valid selection.
  this->EnableSegmentationWidgets(valid);
}

mitk::ToolManager::DataVectorType QmitkMIDASBaseSegmentationFunctionality::GetWorkingNodesFromToolManager()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::ToolManager::DataVectorType result = toolManager->GetWorkingData();
  return result;
}

mitk::Image* QmitkMIDASBaseSegmentationFunctionality::GetWorkingImageFromToolManager(int i)
{
  mitk::Image* result = NULL;

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingNodesFromToolManager();
  if (workingData.size() > 0 && i >= 0 && i < (int)workingData.size())
  {
    mitk::DataNode::Pointer node = workingData[i];

    if (node.IsNotNull())
    {
      mitk::Image::Pointer image = dynamic_cast<mitk::Image*>( node->GetData() );
      if (image.IsNotNull())
      {
        result = image.GetPointer();
      }
    }
  }
  return result;
}

mitk::DataNode* QmitkMIDASBaseSegmentationFunctionality::GetReferenceNodeFromToolManager()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::DataNode::Pointer node = toolManager->GetReferenceData(0);

  return node;
}

mitk::Image* QmitkMIDASBaseSegmentationFunctionality::GetReferenceImageFromToolManager()
{
  mitk::Image* result = NULL;

  mitk::DataNode::Pointer node = this->GetReferenceNodeFromToolManager();
  if (node.IsNotNull())
  {
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>( node->GetData() );
    if (image.IsNotNull())
    {
      result = image.GetPointer();
    }
  }
  return result;
}

mitk::DataNode* QmitkMIDASBaseSegmentationFunctionality::GetReferenceNodeFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  mitk::DataNode* result = FindFirstParentImage(this->GetDefaultDataStorage(), node, false );
  return result;
}

mitk::ToolManager::DataVectorType QmitkMIDASBaseSegmentationFunctionality::GetWorkingNodes()
{
  mitk::ToolManager::DataVectorType result = this->GetWorkingNodesFromToolManager();
  return result;
}

mitk::Image* QmitkMIDASBaseSegmentationFunctionality::GetReferenceImage()
{
  mitk::Image* result = this->GetReferenceImageFromToolManager();
  return result;
}

bool QmitkMIDASBaseSegmentationFunctionality::IsNodeAReferenceImage(const mitk::DataNode::Pointer node)
{
  return IsNodeAGreyScaleImage(node);
}

bool QmitkMIDASBaseSegmentationFunctionality::IsNodeASegmentationImage(const mitk::DataNode::Pointer node)
{
  return IsNodeABinaryImage(node);
}

bool QmitkMIDASBaseSegmentationFunctionality::IsNodeAWorkingImage(const mitk::DataNode::Pointer node)
{
  return IsNodeABinaryImage(node);
}

mitk::ToolManager::DataVectorType QmitkMIDASBaseSegmentationFunctionality::GetWorkingNodesFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  // This default implementation just says Segmentation node == Working node, which subclasses could override.

  mitk::ToolManager::DataVectorType result;
  result.push_back(node);
  return result;
}

mitk::DataNode* QmitkMIDASBaseSegmentationFunctionality::GetSegmentationNodeFromWorkingNode(const mitk::DataNode::Pointer node)
{
  // This default implementation just says Segmentation node == Working node, which subclasses could override.

  mitk::DataNode::Pointer result = node;
  return result;
}

mitk::DataNode* QmitkMIDASBaseSegmentationFunctionality::OnCreateNewSegmentationButtonPressed()
{
  mitk::DataNode::Pointer emptySegmentation = NULL;

  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  // Assumption: If a reference image is selected in the data manager, then it MUST be registered with ToolManager, and hence this is the one we intend to segment.
  mitk::DataNode::Pointer referenceNode = this->GetReferenceNodeFromToolManager();
  if (referenceNode.IsNotNull())
  {
    // Assumption: If a reference image is selected in the data manager, then it MUST be registered with ToolManager, and hence this is the one we intend to segment.
    mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
    if (referenceImage.IsNotNull())
    {
      if (referenceImage->GetDimension() > 2)
      {
        QmitkMIDASNewSegmentationDialog* dialog = new QmitkMIDASNewSegmentationDialog( m_Parent ); // needs a QWidget as parent, "this" is not QWidget
        int dialogReturnValue = dialog->exec();
        if ( dialogReturnValue == QDialog::Rejected ) return NULL; // user clicked cancel or pressed Esc or something similar

        mitk::Tool* firstTool = toolManager->GetToolById(0);
        if (firstTool)
        {
          try
          {
            mitk::Color color = dialog->GetColor();
            emptySegmentation = firstTool->CreateEmptySegmentationNode( referenceImage, dialog->GetSegmentationName().toStdString(), color);
            emptySegmentation->SetColor(color);
            emptySegmentation->SetProperty("binaryimage.selectedcolor", mitk::ColorProperty::New(color));

            if (emptySegmentation.IsNotNull())
            {
              this->ApplyDisplayOptions(emptySegmentation);
              this->GetDefaultDataStorage()->Add(emptySegmentation, referenceNode); // add as a child, because the segmentation "derives" from the original

            } // have got a new segmentation
          }
          catch (std::bad_alloc)
          {
            QMessageBox::warning(NULL,"Create new segmentation","Could not allocate memory for new segmentation");
          }
        } // end if got a tool
      } // end if 3D or above image
      else
      {
        QMessageBox::information(NULL,"Segmentation","Segmentation is currently not supported for 2D images");
      }
    } // end if image not null
    else
    {
      MITK_ERROR << "'Create new segmentation' button should never be clickable unless an image is selected...";
    }
  }
  return emptySegmentation.GetPointer();
}


void QmitkMIDASBaseSegmentationFunctionality::OnComboBoxSelectionChanged( const mitk::DataNode* node )
{
  mitk::DataNode* selectedNode = const_cast<mitk::DataNode*>(node);

  if( selectedNode != NULL )
  {
    m_ImageAndSegmentationSelector->m_SegmentationImagePleaseLoadLabel->hide();
    QmitkMIDASBaseSegmentationFunctionality::OnSelectionChanged( const_cast<mitk::DataNode*>(node) );
  }
  else
  {
    m_ImageAndSegmentationSelector->m_SegmentationImagePleaseLoadLabel->show();
  }
}


void QmitkMIDASBaseSegmentationFunctionality::CreateQtPartControl(QWidget *parent)
{
  if (!m_ImageAndSegmentationSelector)
  {
    m_ImageAndSegmentationSelector = new QmitkMIDASImageAndSegmentationSelectorWidget();
    m_ImageAndSegmentationSelector->setupUi(parent);

    m_ImageAndSegmentationSelector->m_AlignmentWarningLabel->hide();
    m_ImageAndSegmentationSelector->m_SegmentationImagePleaseLoadLabel->setText("<font color='red'>please load an image!</font>");
    m_ImageAndSegmentationSelector->m_SegmentationImageName->hide();
    m_ImageAndSegmentationSelector->m_NewSegmentationButton->setEnabled(false);

    m_ImageAndSegmentationSelector->m_ImageToSegmentComboBox->SetDataStorage(this->GetDefaultDataStorage());
    m_ImageAndSegmentationSelector->m_ImageToSegmentComboBox->SetPredicate(mitk::NodePredicateDataType::New("Image"));

    if( m_ImageAndSegmentationSelector->m_ImageToSegmentComboBox->GetSelectedNode().IsNotNull() )
    {
      m_ImageAndSegmentationSelector->m_SegmentationImagePleaseLoadLabel->hide();
    }
    else
    {
      m_ImageAndSegmentationSelector->m_SegmentationImagePleaseLoadLabel->show();
    }

    mitk::ToolManager* toolManager = this->GetToolManager();
    assert ( toolManager );

    toolManager->SetDataStorage( *(this->GetDefaultDataStorage()) );

    toolManager->NewNodesGenerated +=
      mitk::MessageDelegate<QmitkMIDASBaseSegmentationFunctionality>( this, &QmitkMIDASBaseSegmentationFunctionality::NewNodesGenerated );          // update the list of segmentations
    toolManager->NewNodeObjectsGenerated +=
      mitk::MessageDelegate1<QmitkMIDASBaseSegmentationFunctionality, mitk::ToolManager::DataVectorType*>( this, &QmitkMIDASBaseSegmentationFunctionality::NewNodeObjectsGenerated );          // update the list of segmentations
  }
}


void QmitkMIDASBaseSegmentationFunctionality::CreateConnections()
{
  if (m_ImageAndSegmentationSelector != NULL)
  {
    connect( m_ImageAndSegmentationSelector->m_ImageToSegmentComboBox, SIGNAL( OnSelectionChanged( const mitk::DataNode* ) ), this, SLOT( OnComboBoxSelectionChanged( const mitk::DataNode* ) ) );
  }
}


void QmitkMIDASBaseSegmentationFunctionality::ApplyDisplayOptions(mitk::DataNode* node)
{
  if (!node) return;

  bool isBinary(false);
  if (node->GetBoolProperty("binary", isBinary) && isBinary)
  {
    node->ReplaceProperty("reslice interpolation", mitk::VtkResliceInterpolationProperty::New(VTK_RESLICE_NEAREST), const_cast<const mitk::BaseRenderer*>((mitk::BaseRenderer*)NULL));
    node->SetBoolProperty("outline binary", true);
    node->SetFloatProperty ("outline width", 1.0);
    node->SetBoolProperty("showVolume", false);
    node->SetBoolProperty("volumerendering", false);
    node->SetOpacity(1.0);
  }
}


void QmitkMIDASBaseSegmentationFunctionality::ForceDisplayPreferencesUponAllImages()
{
  if (!m_Parent || !m_Parent->isVisible()) return;

  if (!m_ImageAndSegmentationSelector) return; // might happen on initialization (preferences loaded)

  mitk::DataNode::Pointer referenceData = this->GetReferenceNodeFromToolManager();
  if (referenceData.IsNotNull())
  {
    // iterate all images
    mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();

    mitk::DataStorage::SetOfObjects::ConstPointer allImages = this->GetDefaultDataStorage()->GetSubset( isImage );
    for ( mitk::DataStorage::SetOfObjects::const_iterator iter = allImages->begin(); iter != allImages->end(); ++iter)
    {
      mitk::DataNode* node = *iter;

      // apply display preferences
      ApplyDisplayOptions(node);
    }
  }

  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}


void QmitkMIDASBaseSegmentationFunctionality::SetToolManagerSelection(const mitk::DataNode* referenceData, const mitk::ToolManager::DataVectorType workingDataNodes)
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  toolManager->SetReferenceData(const_cast<mitk::DataNode*>(referenceData));
  toolManager->SetWorkingData(workingDataNodes);

  if (referenceData)
  {
    m_ImageAndSegmentationSelector->m_SegmentationImagePleaseLoadLabel->hide();

    if (workingDataNodes.size() == 0)
    {
      m_ImageAndSegmentationSelector->m_NewSegmentationButton->setEnabled(true);
      m_ImageAndSegmentationSelector->m_WorkingImageSelectionWarningLabel->show();
      m_ImageAndSegmentationSelector->m_SegmentationImageName->hide();
    }
    else
    {
      mitk::DataNode::Pointer node = workingDataNodes[0];
      mitk::DataNode::Pointer segmentationImage = this->GetSegmentationNodeFromWorkingNode(node);
      assert(segmentationImage);

      m_ImageAndSegmentationSelector->m_NewSegmentationButton->setEnabled(false);
      m_ImageAndSegmentationSelector->m_WorkingImageSelectionWarningLabel->hide();
      m_ImageAndSegmentationSelector->m_SegmentationImageName->setText(segmentationImage->GetName().c_str());
      m_ImageAndSegmentationSelector->m_SegmentationImageName->show();
    }
  }
  else
  {
    m_ImageAndSegmentationSelector->m_SegmentationImagePleaseLoadLabel->show();
    m_ImageAndSegmentationSelector->m_NewSegmentationButton->setEnabled(false);
    m_ImageAndSegmentationSelector->m_WorkingImageSelectionWarningLabel->hide();
    m_ImageAndSegmentationSelector->m_SegmentationImageName->hide();
  }
}

template<typename TPixel, unsigned int VImageDimension>
void
QmitkMIDASBaseSegmentationFunctionality
::GetAxisFromITK(
  itk::Image<TPixel, VImageDimension>* itkImage,
  ORIENTATION_ENUM orientation,
  int &outputAxis
  )
{
  outputAxis = -1;

  typename itk::SpatialOrientationAdapter adaptor;
  typename itk::SpatialOrientation::ValidCoordinateOrientationFlags orientationFlag;
  orientationFlag = adaptor.FromDirectionCosines(itkImage->GetDirection());
  std::string orientationString = itk::ConvertSpatialOrientationToString(orientationFlag);

  if (orientationString != "UNKNOWN")
  {
    for (int i = 0; i < 3; i++)
    {
      if (orientation == AXIAL && (orientationString[i] == 'S' || orientationString[i] == 'I'))
      {
        outputAxis = i;
        break;
      }

      if (orientation == CORONAL && (orientationString[i] == 'A' || orientationString[i] == 'P'))
      {
        outputAxis = i;
        break;
      }

      if (orientation == SAGITTAL && (orientationString[i] == 'L' || orientationString[i] == 'R'))
      {
        outputAxis = i;
        break;
      }
    }
  }
}

int QmitkMIDASBaseSegmentationFunctionality::GetAxis(ORIENTATION_ENUM orientation)
{
  int axis = -1;
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    try
    {
      AccessFixedDimensionByItk_n(referenceImage, GetAxisFromITK, 3, (orientation, axis));
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so can't get axis:" << e.what();
    }
  }
  return axis;
}

int QmitkMIDASBaseSegmentationFunctionality::GetAxialAxis()
{
  return this->GetAxis(AXIAL);
}

int QmitkMIDASBaseSegmentationFunctionality::GetCoronalAxis()
{
  return this->GetAxis(CORONAL);
}

int QmitkMIDASBaseSegmentationFunctionality::GetSagittalAxis()
{
  return this->GetAxis(SAGITTAL);
}

void QmitkMIDASBaseSegmentationFunctionality::UpdateVolumeProperty(mitk::DataNode::Pointer segmentationImageNode)
{
  if (segmentationImageNode.IsNotNull())
  {
    mitk::Image::Pointer segmentationImage = dynamic_cast<mitk::Image*>(segmentationImageNode->GetData());
    if (segmentationImage.IsNotNull())
    {
      double segmentationVolume = 0;

      try
      {
        AccessFixedDimensionByItk_n(segmentationImage, GetVolumeFromITK, 3, (segmentationVolume));
      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Caught exception, so can't get axis:" << e.what();
      }

      segmentationImageNode->SetFloatProperty("midas.volume", (float)segmentationVolume);
    }
  }
}

template<typename TPixel, unsigned int VImageDimension>
void
QmitkMIDASBaseSegmentationFunctionality
::GetVolumeFromITK(
    itk::Image<TPixel, VImageDimension>* itkImage,
    double &imageVolume
    )
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::SpacingType SpacingType;

  SpacingType imageSpacing = itkImage->GetSpacing();
  double voxelVolume = 1;
  for ( unsigned int i = 0; i < imageSpacing.Size(); i++)
  {
    voxelVolume *= imageSpacing[i];
  }

  unsigned long int numberOfForegroundVoxels = 0;
  itk::ImageRegionConstIterator<ImageType> iter(itkImage, itkImage->GetLargestPossibleRegion());
  for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter)
  {
    if (iter.Get() > 0)
    {
      numberOfForegroundVoxels++;
    }
  }

  imageVolume = numberOfForegroundVoxels * voxelVolume;
}

void QmitkMIDASBaseSegmentationFunctionality::WipeTools()
{
  mitk::ToolManager::Pointer toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::MIDASTool::Pointer midasTool = dynamic_cast<mitk::MIDASTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASSeedTool>()));
  assert(midasTool);
  midasTool->Wipe();
  midasTool = dynamic_cast<mitk::MIDASTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASPolyTool>()));
  assert(midasTool);
  midasTool->Wipe();
  midasTool = dynamic_cast<mitk::MIDASTool*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::MIDASDrawTool>()));
  assert(midasTool);
  midasTool->Wipe();
}

void QmitkMIDASBaseSegmentationFunctionality::SetReferenceImageSelected()
{
  mitk::DataNode::Pointer referenceDataNode = this->GetReferenceNodeFromToolManager();
  this->FireNodeSelected(referenceDataNode);
}
