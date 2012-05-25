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
#include "mitkBaseRenderer.h"
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
#include "QmitkMIDASMultiViewWidget.h"
#include "NifTKConfigure.h"
#include "itkMIDASHelper.h"

const std::string QmitkMIDASBaseSegmentationFunctionality::DEFAULT_COLOUR("midas editor default colour");
const std::string QmitkMIDASBaseSegmentationFunctionality::DEFAULT_COLOUR_STYLE_SHEET("midas editor default colour style sheet");

QmitkMIDASBaseSegmentationFunctionality::QmitkMIDASBaseSegmentationFunctionality()
:
  m_SelectedNode(NULL)
, m_SelectedImage(NULL)
, m_ImageAndSegmentationSelector(NULL)
, m_ToolSelector(NULL)
{
  m_SelectedNode = NULL;
}

QmitkMIDASBaseSegmentationFunctionality::~QmitkMIDASBaseSegmentationFunctionality()
{
  if (m_ImageAndSegmentationSelector != NULL)
  {
    delete m_ImageAndSegmentationSelector;
  }

  if (m_ToolSelector != NULL)
  {
    delete m_ToolSelector;
  }
}

QmitkMIDASBaseSegmentationFunctionality::QmitkMIDASBaseSegmentationFunctionality(
    const QmitkMIDASBaseSegmentationFunctionality& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

void QmitkMIDASBaseSegmentationFunctionality::CreateQtPartControl(QWidget *parentForSelectorWidget, QWidget *parentForToolWidget)
{
  if (!m_ImageAndSegmentationSelector)
  {
    // Set up the Image and Segmentation Selector.
    // Subclasses add it to their layouts, at the appropriate point.
    m_ImageAndSegmentationSelector = new QmitkMIDASImageAndSegmentationSelectorWidget(parentForSelectorWidget);
    m_ImageAndSegmentationSelector->m_AlignmentWarningLabel->hide();
    m_ImageAndSegmentationSelector->m_SegmentationImagePleaseLoadLabel->setText("<font color='red'>please load an image!</font>");
    m_ImageAndSegmentationSelector->m_SegmentationImageName->hide();
    m_ImageAndSegmentationSelector->m_NewSegmentationButton->setEnabled(false);
    m_ImageAndSegmentationSelector->m_ImageToSegmentComboBox->SetDataStorage(this->GetDataStorage());
    m_ImageAndSegmentationSelector->m_ImageToSegmentComboBox->SetPredicate(mitk::NodePredicateDataType::New("Image"));
    if( m_ImageAndSegmentationSelector->m_ImageToSegmentComboBox->GetSelectedNode().IsNotNull() )
    {
      m_ImageAndSegmentationSelector->m_SegmentationImagePleaseLoadLabel->hide();
    }
    else
    {
      m_ImageAndSegmentationSelector->m_SegmentationImagePleaseLoadLabel->show();
    }

    // Set up the Tool Selector.
    // Subclasses add it to their layouts, at the appropriate point.
    m_ToolSelector = new QmitkMIDASToolSelectorWidget(parentForToolWidget);
    m_ToolSelector->m_ManualToolSelectionBox->SetGenerateAccelerators(true);
    m_ToolSelector->m_ManualToolSelectionBox->SetLayoutColumns(3);
    m_ToolSelector->m_ManualToolSelectionBox->SetToolGUIArea( m_ToolSelector->m_ManualToolGUIContainer );
    m_ToolSelector->m_ManualToolSelectionBox->SetEnabledMode( QmitkToolSelectionBox::EnabledWithReferenceAndWorkingData );

    this->RetrievePreferenceValues();

    // Connect the ToolManager to DataStorage straight away.
    mitk::ToolManager* toolManager = this->GetToolManager();
    assert ( toolManager );
    toolManager->SetDataStorage( *(this->GetDataStorage()) );

    // We listen to NewNodesGenerated messages as the ToolManager is responsible for instantiating them.
    toolManager->NewNodesGenerated +=
      mitk::MessageDelegate<QmitkMIDASBaseSegmentationFunctionality>( this, &QmitkMIDASBaseSegmentationFunctionality::NewNodesGenerated );

    // We listen to NewNodObjectGenerated as the ToolManager is responsible for instantiating them.
    toolManager->NewNodeObjectsGenerated +=
      mitk::MessageDelegate1<QmitkMIDASBaseSegmentationFunctionality, mitk::ToolManager::DataVectorType*>( this, &QmitkMIDASBaseSegmentationFunctionality::NewNodeObjectsGenerated );
  }
}

mitk::ToolManager* QmitkMIDASBaseSegmentationFunctionality::GetToolManager()
{
  return m_ToolSelector->m_ManualToolSelectionBox->GetToolManager();
}

void QmitkMIDASBaseSegmentationFunctionality::OnToolSelected(int toolID)
{
  if (this->GetActiveStdMultiWidget() != NULL && this->GetActiveMIDASMultiViewWidget() != NULL)
  {
    if (toolID >= 0)
    {
      this->GetActiveStdMultiWidget()->GetMouseModeSwitcher()->SetInteractionScheme(mitk::MouseModeSwitcher::OFF);
      this->GetActiveStdMultiWidget()->DisableNavigationControllerEventListening();
      this->GetActiveMIDASMultiViewWidget()->SetNavigationControllerEventListening(false);
    }
    else
    {
      this->GetActiveStdMultiWidget()->GetMouseModeSwitcher()->SetInteractionScheme(mitk::MouseModeSwitcher::MITK);
      this->GetActiveStdMultiWidget()->EnableNavigationControllerEventListening();
      this->GetActiveMIDASMultiViewWidget()->SetNavigationControllerEventListening(true);
    }
  }
}

void QmitkMIDASBaseSegmentationFunctionality::NewNodesGenerated()
{
  std::cerr << "Matt, not finished yet, QmitkMIDASBaseSegmentationFunctionality::NewNodesGenerated" << std::endl;
}

void QmitkMIDASBaseSegmentationFunctionality::NewNodeObjectsGenerated(mitk::ToolManager::DataVectorType* nodes)
{
  std::cerr << "Matt, not finished yet, QmitkMIDASBaseSegmentationFunctionality::NewNodeObjectsGenerated" << std::endl;
}

void QmitkMIDASBaseSegmentationFunctionality::SelectNode(const mitk::DataNode::Pointer node)
{
  assert(node);
  this->FireNodeSelected(node);
}

void QmitkMIDASBaseSegmentationFunctionality::OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes)
{
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
  mitk::DataNode* result = FindFirstParentImage(this->GetDataStorage(), node, false );
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

mitk::DataNode* QmitkMIDASBaseSegmentationFunctionality::OnCreateNewSegmentationButtonPressed(QColor &defaultColor)
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
        QmitkMIDASNewSegmentationDialog* dialog = new QmitkMIDASNewSegmentationDialog(defaultColor, m_Parent ); // needs a QWidget as parent, "this" is not QWidget
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
              this->GetDataStorage()->Add(emptySegmentation, referenceNode); // add as a child, because the segmentation "derives" from the original

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
    this->SelectNode(selectedNode);
  }
  else
  {
    m_ImageAndSegmentationSelector->m_SegmentationImagePleaseLoadLabel->show();
  }
}


void QmitkMIDASBaseSegmentationFunctionality::CreateConnections()
{
  if (m_ImageAndSegmentationSelector != NULL)
  {
    connect( m_ImageAndSegmentationSelector->m_ImageToSegmentComboBox, SIGNAL( OnSelectionChanged( const mitk::DataNode* ) ), this, SLOT( OnComboBoxSelectionChanged( const mitk::DataNode* ) ) );
  }
}

void QmitkMIDASBaseSegmentationFunctionality::SetEnableManualToolSelectionBox(bool enabled)
{
  this->m_ToolSelector->m_ManualToolSelectionBox->QWidget::setEnabled(enabled);
  this->m_ToolSelector->m_ManualToolGUIContainer->setEnabled(enabled);
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

    mitk::DataStorage::SetOfObjects::ConstPointer allImages = this->GetDataStorage()->GetSubset( isImage );
    for ( mitk::DataStorage::SetOfObjects::const_iterator iter = allImages->begin(); iter != allImages->end(); ++iter)
    {
      mitk::DataNode* node = *iter;

      // apply display preferences
      ApplyDisplayOptions(node);
    }
  }
  QmitkAbstractView::RequestRenderWindowUpdate();
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

int QmitkMIDASBaseSegmentationFunctionality::GetSliceNumberFromSliceNavigationControllerAndReferenceImage()
{
  int sliceNumber = -1;

  mitk::SliceNavigationController::Pointer snc = this->GetSliceNavigationController();
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();

  if (referenceImage.IsNotNull() && snc.IsNotNull())
  {
    mitk::PlaneGeometry::ConstPointer pg = snc->GetCurrentPlaneGeometry();
    if (pg.IsNotNull())
    {
      mitk::Point3D originInMillimetres = pg->GetOrigin();
      mitk::Point3D originInVoxelCoordinates;
      referenceImage->GetGeometry()->WorldToIndex(originInMillimetres, originInVoxelCoordinates);

      int viewAxis = this->GetViewAxis();
      sliceNumber = (int)(originInVoxelCoordinates[viewAxis] + 0.5);
    }
  }
  return sliceNumber;
}

itk::ORIENTATION_ENUM QmitkMIDASBaseSegmentationFunctionality::GetOrientationAsEnum()
{
  itk::ORIENTATION_ENUM orientation = itk::ORIENTATION_UNKNOWN;
  mitk::SliceNavigationController* sliceNavigationController = this->GetSliceNavigationController();
  if (sliceNavigationController != NULL)
  {
    mitk::SliceNavigationController::ViewDirection viewDirection = sliceNavigationController->GetViewDirection();

    if (viewDirection == mitk::SliceNavigationController::Transversal)
    {
      orientation = itk::ORIENTATION_AXIAL;
    }
    else if (viewDirection == mitk::SliceNavigationController::Sagittal)
    {
      orientation = itk::ORIENTATION_SAGITTAL;
    }
    else if (viewDirection == mitk::SliceNavigationController::Frontal)
    {
      orientation = itk::ORIENTATION_CORONAL;
    }
  }
  return orientation;
}

int QmitkMIDASBaseSegmentationFunctionality::GetAxisFromReferenceImage(itk::ORIENTATION_ENUM orientation)
{
  int axis = -1;
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    try
    {
      AccessFixedDimensionByItk_n(referenceImage, GetAxisFromReferenceImageUsingITK, 3, (orientation, axis));
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so can't get axis:" << e.what();
    }
  }
  return axis;
}

template<typename TPixel, unsigned int VImageDimension>
void
QmitkMIDASBaseSegmentationFunctionality
::GetAxisFromReferenceImageUsingITK(
  itk::Image<TPixel, VImageDimension>* itkImage,
  itk::ORIENTATION_ENUM orientation,
  int &outputAxis
  )
{
  itk::GetAxisFromITKImage(itkImage, orientation, outputAxis);
}

int QmitkMIDASBaseSegmentationFunctionality::GetReferenceImageAxialAxis()
{
  return this->GetAxisFromReferenceImage(itk::ORIENTATION_AXIAL);
}

int QmitkMIDASBaseSegmentationFunctionality::GetReferenceImageCoronalAxis()
{
  return this->GetAxisFromReferenceImage(itk::ORIENTATION_CORONAL);
}

int QmitkMIDASBaseSegmentationFunctionality::GetReferenceImageSagittalAxis()
{
  return this->GetAxisFromReferenceImage(itk::ORIENTATION_SAGITTAL);
}

int QmitkMIDASBaseSegmentationFunctionality::GetViewAxis()
{

  int axisNumber = -1;

  // Use the above method to work out which orientation we are currently looking at, in the current 2D window.
  itk::ORIENTATION_ENUM orientation = this->GetOrientationAsEnum();
  if (orientation != -1)
  {
    axisNumber = this->GetAxisFromReferenceImage(orientation);
  }

  return axisNumber;
}

int QmitkMIDASBaseSegmentationFunctionality::GetUpDirection()
{
  int upDirection = 0;

  itk::ORIENTATION_ENUM orientation = this->GetOrientationAsEnum();
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    try
    {
      AccessFixedDimensionByItk_n(referenceImage, GetUpDirectionUsingITK, 3, (orientation, upDirection));
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so can't get up direction:" << e.what();
    }
  }
  return upDirection;
}

template<typename TPixel, unsigned int VImageDimension>
void
QmitkMIDASBaseSegmentationFunctionality
::GetUpDirectionUsingITK(
    itk::Image<TPixel, VImageDimension>* itkImage,
    itk::ORIENTATION_ENUM orientation,
    int &upDirection
)
{
  GetUpDirectionFromITKImage(itkImage, orientation, upDirection);
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
  itk::GetVolumeFromITKImage(itkImage, imageVolume);
}

void QmitkMIDASBaseSegmentationFunctionality::SetReferenceImageSelected()
{
  mitk::DataNode::Pointer referenceDataNode = this->GetReferenceNodeFromToolManager();
  this->FireNodeSelected(referenceDataNode);
}

void QmitkMIDASBaseSegmentationFunctionality::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}

void QmitkMIDASBaseSegmentationFunctionality::RetrievePreferenceValues()
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  assert( prefService );

  berry::IBerryPreferences::Pointer prefs
      = (prefService->GetSystemPreferences()->Node(this->GetPreferencesNodeName()))
        .Cast<berry::IBerryPreferences>();

  assert( prefs );

  QString defaultColorName = QString::fromStdString (prefs->GetByteArray(QmitkMIDASBaseSegmentationFunctionality::DEFAULT_COLOUR, ""));
  m_DefaultSegmentationColor = QColor(defaultColorName);
  if (defaultColorName=="") // default values
  {
    m_DefaultSegmentationColor = QColor(0, 255, 0);
  }
}

