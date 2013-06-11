/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASBaseSegmentationFunctionality.h"

#include <QMessageBox>
#include "internal/MIDASActivator.h"
#include <mitkILinkedRenderWindowPart.h>
#include <mitkImageAccessByItk.h>
#include <mitkDataNodeObject.h>
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateProperty.h>
#include <mitkNodePredicateAnd.h>
#include <mitkNodePredicateNot.h>
#include <mitkProperties.h>
#include <mitkColorProperty.h>
#include <mitkRenderingManager.h>
#include <mitkBaseRenderer.h>
#include <mitkSegTool2D.h>
#include <mitkVtkResliceInterpolationProperty.h>
#include <mitkToolManager.h>
#include <mitkGlobalInteraction.h>
#include <mitkStateMachine.h>
#include <mitkDataStorageUtils.h>
#include <mitkColorProperty.h>
#include <mitkProperties.h>
#include <QmitkRenderWindow.h>

#include <NifTKConfigure.h>
#include <QmitkMIDASNewSegmentationDialog.h>
#include <mitkMIDASTool.h>
#include <mitkMIDASDrawTool.h>
#include <mitkMIDASPolyTool.h>
#include <mitkMIDASSeedTool.h>
#include <mitkMIDASOrientationUtils.h>
#include <itkMIDASHelper.h>

const std::string QmitkMIDASBaseSegmentationFunctionality::DEFAULT_COLOUR("midas editor default colour");
const std::string QmitkMIDASBaseSegmentationFunctionality::DEFAULT_COLOUR_STYLE_SHEET("midas editor default colour style sheet");


//-----------------------------------------------------------------------------
QmitkMIDASBaseSegmentationFunctionality::QmitkMIDASBaseSegmentationFunctionality()
:
  m_SelectedNode(NULL)
, m_SelectedImage(NULL)
, m_ImageAndSegmentationSelector(NULL)
, m_ToolSelector(NULL)
, m_SegmentationView(NULL)
, m_Context(NULL)
, m_EventAdmin(NULL)
{
  m_SelectedNode = NULL;
}


//-----------------------------------------------------------------------------
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

  m_SegmentationView->Deactivated();

  if (m_SegmentationView != NULL)
  {
    delete m_SegmentationView;
  }

}


//-----------------------------------------------------------------------------
void QmitkMIDASBaseSegmentationFunctionality::Activated()
{
  QmitkBaseView::Activated();

  berry::IWorkbenchPart::Pointer nullPart;
  this->OnSelectionChanged(nullPart, this->GetDataManagerSelection());
}


//-----------------------------------------------------------------------------
void QmitkMIDASBaseSegmentationFunctionality::Visible()
{
  QmitkBaseView::Visible();
}


//-----------------------------------------------------------------------------
void QmitkMIDASBaseSegmentationFunctionality::Hidden()
{
  QmitkBaseView::Hidden();
}


//-----------------------------------------------------------------------------
QmitkMIDASBaseSegmentationFunctionality::QmitkMIDASBaseSegmentationFunctionality(
    const QmitkMIDASBaseSegmentationFunctionality& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
void QmitkMIDASBaseSegmentationFunctionality::CreateQtPartControl(QWidget *parent)
{
  if (!m_ImageAndSegmentationSelector)
  {

    // Set up the Image and Segmentation Selector.
    // Subclasses add it to their layouts, at the appropriate point.
    m_ContainerForSelectorWidget = new QWidget(parent);
    m_ImageAndSegmentationSelector = new QmitkMIDASImageAndSegmentationSelectorWidget(m_ContainerForSelectorWidget);
    m_ImageAndSegmentationSelector->m_NewSegmentationButton->setEnabled(false);
    m_ImageAndSegmentationSelector->m_ReferenceImageNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
    m_ImageAndSegmentationSelector->m_ReferenceImageNameLabel->show();
    m_ImageAndSegmentationSelector->m_SegmentationImageNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
    m_ImageAndSegmentationSelector->m_SegmentationImageNameLabel->show();

    // Set up the Tool Selector.
    // Subclasses add it to their layouts, at the appropriate point.
    m_ContainerForToolWidget = new QWidget(parent);
    m_ToolSelector = new QmitkMIDASToolSelectorWidget(m_ContainerForToolWidget);
    m_ToolSelector->m_ManualToolSelectionBox->SetGenerateAccelerators(true);
    m_ToolSelector->m_ManualToolSelectionBox->SetLayoutColumns(3);
    m_ToolSelector->m_ManualToolSelectionBox->SetToolGUIArea( m_ToolSelector->m_ManualToolGUIContainer );
    m_ToolSelector->m_ManualToolSelectionBox->SetEnabledMode( QmitkToolSelectionBox::EnabledWithReferenceAndWorkingData );

    // Set up the Segmentation View
    // Subclasses add it to their layouts, at the appropriate point.
    m_ContainerForSegmentationViewWidget = new QWidget(parent);
    m_SegmentationView = new QmitkMIDASSegmentationViewWidget(m_ContainerForSegmentationViewWidget);
    m_SegmentationView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_SegmentationView->SetDataStorage(this->GetDataStorage());
    m_SegmentationView->SetContainingFunctionality(this);
    m_SegmentationView->Activated();

    // Retrieving preferences done in another method so we can call it on startup, and when prefs change.
    this->RetrievePreferenceValues();

    // Connect the ToolManager to DataStorage straight away.
    mitk::ToolManager* toolManager = this->GetToolManager();
    assert ( toolManager );
    toolManager->SetDataStorage( *(this->GetDataStorage()) );

    // Set up the ctkEventAdmin stuff.
    m_Context = mitk::MIDASActivator::GetPluginContext();
    m_EventAdminRef = m_Context->getServiceReference<ctkEventAdmin>();
    m_EventAdmin = m_Context->getService<ctkEventAdmin>(m_EventAdminRef);
    m_EventAdmin->publishSignal(this, SIGNAL(InteractorRequest(ctkDictionary)),
                              "org/mitk/gui/qt/INTERACTOR_REQUEST", Qt::QueuedConnection);
  }
}


//-----------------------------------------------------------------------------
mitk::ToolManager* QmitkMIDASBaseSegmentationFunctionality::GetToolManager()
{
  return m_ToolSelector->m_ManualToolSelectionBox->GetToolManager();
}


//-----------------------------------------------------------------------------
void QmitkMIDASBaseSegmentationFunctionality::OnToolSelected(int toolID)
{
  // See http://bugs.mitk.org/show_bug.cgi?id=12302 - new interaction concept.
  if (toolID >= 0)
  {
    // Enabling a tool - so just the tool receives the event,
    // so tool must return a high value from mitk::CanHandleEvent
    mitk::GlobalInteraction::GetInstance()->SetEventNotificationPolicy(mitk::GlobalInteraction::INFORM_ONE);
  }
  else
  {
    // Disabling a tool - revert to default behaviour.
    mitk::GlobalInteraction::GetInstance()->SetEventNotificationPolicy(mitk::GlobalInteraction::INFORM_MULTIPLE);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASBaseSegmentationFunctionality::OnSelectionChanged(berry::IWorkbenchPart::Pointer /*part*/, const QList<mitk::DataNode::Pointer> &nodes)
{
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

    if (mitk::IsNodeABinaryImage(node) &&
        this->CanStartSegmentationForBinaryNode(node) &&
        !this->IsNodeASegmentationImage(node))
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
    m_ImageAndSegmentationSelector->m_ReferenceImageNameLabel->blockSignals(true);
    if (referenceData.IsNotNull())
    {

      m_ImageAndSegmentationSelector->m_ReferenceImageNameLabel->setText(tr("<font color='black'>%1</font>").arg(referenceData->GetName().c_str()));
    }
    else
    {
      m_ImageAndSegmentationSelector->m_ReferenceImageNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
    }
    m_ImageAndSegmentationSelector->m_ReferenceImageNameLabel->blockSignals(false);

    // Tell the tool manager the images for reference and working purposes.
    this->SetToolManagerSelection(referenceData, workingDataNodes);

  }

  // Adjust widgets according to whether we have a valid selection.
  this->EnableSegmentationWidgets(valid);
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType QmitkMIDASBaseSegmentationFunctionality::GetWorkingNodesFromToolManager()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::ToolManager::DataVectorType result = toolManager->GetWorkingData();
  return result;
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
mitk::DataNode* QmitkMIDASBaseSegmentationFunctionality::GetReferenceNodeFromToolManager()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  mitk::DataNode::Pointer node = toolManager->GetReferenceData(0);

  return node;
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
mitk::DataNode* QmitkMIDASBaseSegmentationFunctionality::GetReferenceNodeFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  mitk::DataNode* result = FindFirstParentImage(this->GetDataStorage(), node, false );
  return result;
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType QmitkMIDASBaseSegmentationFunctionality::GetWorkingNodes()
{
  mitk::ToolManager::DataVectorType result = this->GetWorkingNodesFromToolManager();
  return result;
}


//-----------------------------------------------------------------------------
mitk::Image* QmitkMIDASBaseSegmentationFunctionality::GetReferenceImage()
{
  mitk::Image* result = this->GetReferenceImageFromToolManager();
  return result;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASBaseSegmentationFunctionality::IsNodeAReferenceImage(const mitk::DataNode::Pointer node)
{
  return IsNodeAGreyScaleImage(node);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASBaseSegmentationFunctionality::IsNodeASegmentationImage(const mitk::DataNode::Pointer node)
{
  return IsNodeABinaryImage(node);
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
mitk::DataNode* QmitkMIDASBaseSegmentationFunctionality::GetSegmentationNodeFromWorkingNode(const mitk::DataNode::Pointer node)
{
  // This default implementation just says Segmentation node == Working node, which subclasses could override.

  mitk::DataNode::Pointer result = node;
  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* QmitkMIDASBaseSegmentationFunctionality::CreateNewSegmentation(QColor &defaultColor)
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
        QmitkMIDASNewSegmentationDialog* dialog = new QmitkMIDASNewSegmentationDialog(defaultColor, this->GetParent() ); // needs a QWidget as parent, "this" is not QWidget
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
            emptySegmentation->SetProperty("midas.tmp.selectedcolor", mitk::ColorProperty::New(color));

            if (emptySegmentation.IsNotNull())
            {
              this->ApplyDisplayOptions(emptySegmentation);
              this->GetDataStorage()->Add(emptySegmentation, referenceNode); // add as a child, because the segmentation "derives" from the original
            } // have got a new segmentation
          }
          catch (std::bad_alloc&)
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


//-----------------------------------------------------------------------------
void QmitkMIDASBaseSegmentationFunctionality::CreateConnections()
{
}


//-----------------------------------------------------------------------------
void QmitkMIDASBaseSegmentationFunctionality::SetEnableManualToolSelectionBox(bool enabled)
{
  m_ToolSelector->m_ManualToolSelectionBox->QWidget::setEnabled(enabled);
  m_ToolSelector->m_ManualToolGUIContainer->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
void QmitkMIDASBaseSegmentationFunctionality::SetToolManagerSelection(const mitk::DataNode* referenceData, const mitk::ToolManager::DataVectorType workingDataNodes)
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  if (workingDataNodes.size() == 0 ||
      ( toolManager->GetWorkingData().size() > 0 &&
        workingDataNodes.size() > 0 &&
        toolManager->GetWorkingData(0) != workingDataNodes[0] ))
  {
    toolManager->ActivateTool(-1);
  }

  toolManager->SetReferenceData(const_cast<mitk::DataNode*>(referenceData));
  toolManager->SetWorkingData(workingDataNodes);

  if (referenceData)
  {
    if (workingDataNodes.size() == 0)
    {
      m_ImageAndSegmentationSelector->m_NewSegmentationButton->setEnabled(true);
      m_ImageAndSegmentationSelector->m_SegmentationImageNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
    }
    else
    {
      mitk::DataNode::Pointer node = workingDataNodes[0];
      mitk::DataNode::Pointer segmentationImage = this->GetSegmentationNodeFromWorkingNode(node);
      assert(segmentationImage);

      m_ImageAndSegmentationSelector->m_NewSegmentationButton->setEnabled(false);
      m_ImageAndSegmentationSelector->m_SegmentationImageNameLabel->setText(tr("<font color='black'>%1</font>").arg(segmentationImage->GetName().c_str()));
    }
  }
  else
  {
    m_ImageAndSegmentationSelector->m_NewSegmentationButton->setEnabled(false);
    m_ImageAndSegmentationSelector->m_SegmentationImageNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
  }
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
MIDASOrientation QmitkMIDASBaseSegmentationFunctionality::GetOrientationAsEnum()
{
  MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;
  mitk::SliceNavigationController* sliceNavigationController = this->GetSliceNavigationController();
  if (sliceNavigationController != NULL)
  {
    mitk::SliceNavigationController::ViewDirection viewDirection = sliceNavigationController->GetViewDirection();

    if (viewDirection == mitk::SliceNavigationController::Axial)
    {
      orientation = MIDAS_ORIENTATION_AXIAL;
    }
    else if (viewDirection == mitk::SliceNavigationController::Sagittal)
    {
      orientation = MIDAS_ORIENTATION_SAGITTAL;
    }
    else if (viewDirection == mitk::SliceNavigationController::Frontal)
    {
      orientation = MIDAS_ORIENTATION_CORONAL;
    }
  }
  return orientation;
}


//-----------------------------------------------------------------------------
int QmitkMIDASBaseSegmentationFunctionality::GetAxisFromReferenceImage(const MIDASOrientation& orientation)
{
  int axis = -1;
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    axis = mitk::GetThroughPlaneAxis(referenceImage, orientation);
  }
  return axis;
}


//-----------------------------------------------------------------------------
int QmitkMIDASBaseSegmentationFunctionality::GetReferenceImageAxialAxis()
{
  return this->GetAxisFromReferenceImage(MIDAS_ORIENTATION_AXIAL);
}


//-----------------------------------------------------------------------------
int QmitkMIDASBaseSegmentationFunctionality::GetReferenceImageCoronalAxis()
{
  return this->GetAxisFromReferenceImage(MIDAS_ORIENTATION_CORONAL);
}


//-----------------------------------------------------------------------------
int QmitkMIDASBaseSegmentationFunctionality::GetReferenceImageSagittalAxis()
{
  return this->GetAxisFromReferenceImage(MIDAS_ORIENTATION_SAGITTAL);
}


//-----------------------------------------------------------------------------
int QmitkMIDASBaseSegmentationFunctionality::GetViewAxis()
{
  int axisNumber = -1;
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  MIDASOrientation orientation = this->GetOrientationAsEnum();
  if (referenceImage.IsNotNull() && orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    axisNumber = mitk::GetThroughPlaneAxis(referenceImage, orientation);
  }
  return axisNumber;
}


//-----------------------------------------------------------------------------
int QmitkMIDASBaseSegmentationFunctionality::GetUpDirection()
{
  int upDirection = 0;
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  MIDASOrientation orientation = this->GetOrientationAsEnum();
  if (referenceImage.IsNotNull() && orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    upDirection = mitk::GetUpDirection(referenceImage, orientation);
  }
  return upDirection;
}


//-----------------------------------------------------------------------------
void QmitkMIDASBaseSegmentationFunctionality::SetReferenceImageSelected()
{
  mitk::DataNode::Pointer referenceImageNode = this->GetReferenceNodeFromToolManager();
  this->SetCurrentSelection(referenceImageNode);
}


//-----------------------------------------------------------------------------
void QmitkMIDASBaseSegmentationFunctionality::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
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
  if (defaultColorName == "") // default values
  {
    m_DefaultSegmentationColor = QColor(0, 255, 0);
  }
}
