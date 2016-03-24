/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseSegmentationView.h"

#include <berryPlatform.h>

#include <QMessageBox>
#include "internal/MIDASActivator.h"
#include <mitkILinkedRenderWindowPart.h>
#include <mitkImageAccessByItk.h>
#include <mitkDataNodeObject.h>
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
#include <niftkBaseSegmentorWidget.h>
#include <niftkMIDASNewSegmentationDialog.h>
#include <niftkMIDASTool.h>
#include <niftkMIDASDrawTool.h>
#include <niftkMIDASPolyTool.h>
#include <niftkMIDASSeedTool.h>
#include <niftkMIDASOrientationUtils.h>

const QString niftkBaseSegmentationView::DEFAULT_COLOUR("midas editor default colour");
const QString niftkBaseSegmentationView::DEFAULT_COLOUR_STYLE_SHEET("midas editor default colour style sheet");


//-----------------------------------------------------------------------------
niftkBaseSegmentationView::niftkBaseSegmentationView()
  : m_SelectedNode(nullptr),
    m_SelectedImage(nullptr),
    m_ActiveToolID(-1),
    m_MainWindowCursorVisibleWithToolsOff(true),
    m_BaseSegmentationViewControls(nullptr)
{
  m_SelectedNode = NULL;
}


//-----------------------------------------------------------------------------
niftkBaseSegmentationView::~niftkBaseSegmentationView()
{
  mitk::ToolManager::ToolVectorTypeConst tools = this->GetToolManager()->GetTools();
  mitk::ToolManager::ToolVectorTypeConst::iterator it = tools.begin();
  for ( ; it != tools.end(); ++it)
  {
    mitk::Tool* tool = const_cast<mitk::Tool*>(it->GetPointer());
    if (niftk::MIDASStateMachine* midasSM = dynamic_cast<niftk::MIDASStateMachine*>(tool))
    {
      midasSM->RemoveEventFilter(this);
    }
  }

  delete m_BaseSegmentationViewControls;
  m_BaseSegmentationViewControls = nullptr;
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentationView::EventFilter(const mitk::StateEvent* stateEvent) const
{
  // If we have a render window part (aka. editor or display)...
  if (mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart())
  {
    // and it has a focused render window...
    if (QmitkRenderWindow* renderWindow = renderWindowPart->GetActiveQmitkRenderWindow())
    {
      // whose renderer is the sender of this event...
      if (renderWindow->GetRenderer() == stateEvent->GetEvent()->GetSender())
      {
        // then we let the event pass through.
        return false;
      }
    }
  }

  // Otherwise, if it comes from another window, we reject it.
  return true;
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentationView::EventFilter(mitk::InteractionEvent* event) const
{
  // If we have a render window part (aka. editor or display)...
  if (mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart())
  {
    // and it has a focused render window...
    if (QmitkRenderWindow* renderWindow = renderWindowPart->GetActiveQmitkRenderWindow())
    {
      // whose renderer is the sender of this event...
      if (renderWindow->GetRenderer() == event->GetSender())
      {
        // then we let the event pass through.
        return false;
      }
    }
  }

  // Otherwise, if it comes from another window, we reject it.
  return true;
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentationView::Activated()
{
  QmitkBaseView::Activated();

  berry::IWorkbenchPart::Pointer nullPart;
  this->OnSelectionChanged(nullPart, this->GetDataManagerSelection());
}


//-----------------------------------------------------------------------------
niftkBaseSegmentationView::niftkBaseSegmentationView(const niftkBaseSegmentationView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentationView::RegisterTools(mitk::ToolManager::Pointer toolManager)
{
  Q_UNUSED(toolManager);
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentationView::CreateQtPartControl(QWidget *parent)
{
  assert(!m_BaseSegmentationViewControls);

  this->SetParent(parent);

  // Create an own tool manager and connect it to the data storage straight away.
  m_ToolManager = mitk::ToolManager::New(this->GetDataStorage());

  this->RegisterTools(m_ToolManager);

  mitk::ToolManager::ToolVectorTypeConst tools = m_ToolManager->GetTools();
  mitk::ToolManager::ToolVectorTypeConst::iterator it = tools.begin();
  for ( ; it != tools.end(); ++it)
  {
    mitk::Tool* tool = const_cast<mitk::Tool*>(it->GetPointer());
    if (niftk::MIDASStateMachine* midasSM = dynamic_cast<niftk::MIDASStateMachine*>(tool))
    {
      midasSM->InstallEventFilter(this);
    }
  }

  // Retrieving preferences done in another method so we can call it on startup, and when prefs change.
  this->RetrievePreferenceValues();

//  m_BaseSegmentationViewControls = new niftkBaseSegmentationViewControls(parent);

//  m_BaseSegmentationViewControls->SetToolManager(m_ToolManager);
}


//-----------------------------------------------------------------------------
mitk::ToolManager* niftkBaseSegmentationView::GetToolManager()
{
  return m_ToolManager;
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentationView::OnToolSelected(int toolID)
{
  if (toolID != -1)
  {
    bool mainWindowCursorWasVisible = this->SetMainWindowCursorVisible(false);

    if (m_ActiveToolID == -1)
    {
      m_MainWindowCursorVisibleWithToolsOff = mainWindowCursorWasVisible;
    }
  }
  else
  {
    this->SetMainWindowCursorVisible(m_MainWindowCursorVisibleWithToolsOff);
  }

  m_ActiveToolID = toolID;

  /// Set the focus back to the main window. This is needed so that the keyboard shortcuts
  /// (like 'a' and 'z' for changing slice) keep on working.
  if (QmitkRenderWindow* mainWindow = this->GetSelectedRenderWindow())
  {
    mainWindow->setFocus();
  }
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentationView::OnSelectionChanged(berry::IWorkbenchPart::Pointer /*part*/, const QList<mitk::DataNode::Pointer> &nodes)
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
    else if (mitk::IsNodeABinaryImage(node) && this->CanStartSegmentationForBinaryNode(node))
    {
      segmentedData = node;
    }

    if (segmentedData.IsNotNull())
    {

      referenceData = this->GetReferenceNodeFromSegmentationNode(segmentedData);

      if (this->IsNodeASegmentationImage(node))
      {
        workingDataNodes = this->GetWorkingDataFromSegmentationNode(segmentedData);
        valid = true;
      }
    }

    // If we have worked out the reference data, then set the combo box.
    m_BaseSegmentationViewControls->m_SegmentationSelectorWidget->m_ReferenceImageNameLabel->blockSignals(true);
    if (referenceData.IsNotNull())
    {
      m_BaseSegmentationViewControls->m_SegmentationSelectorWidget->m_ReferenceImageNameLabel->setText(tr("<font color='black'>%1</font>").arg(referenceData->GetName().c_str()));
    }
    else
    {
      m_BaseSegmentationViewControls->m_SegmentationSelectorWidget->m_ReferenceImageNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
    }
    m_BaseSegmentationViewControls->m_SegmentationSelectorWidget->m_ReferenceImageNameLabel->blockSignals(false);

    // Tell the tool manager the images for reference and working purposes.
    this->SetToolManagerSelection(referenceData, workingDataNodes);

  }

  // Adjust widgets according to whether we have a valid selection.
  m_BaseSegmentationViewControls->EnableSegmentationWidgets(valid);
}


//-----------------------------------------------------------------------------
mitk::Image* niftkBaseSegmentationView::GetWorkingImageFromToolManager(int i)
{
  mitk::Image* result = NULL;

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
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
mitk::DataNode* niftkBaseSegmentationView::GetReferenceNodeFromToolManager()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  return toolManager->GetReferenceData(0);
}


//-----------------------------------------------------------------------------
mitk::Image* niftkBaseSegmentationView::GetReferenceImageFromToolManager()
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
mitk::DataNode* niftkBaseSegmentationView::GetReferenceNodeFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  mitk::DataNode* result = mitk::FindFirstParentImage(this->GetDataStorage(), node, false );
  return result;
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType niftkBaseSegmentationView::GetWorkingData()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  return toolManager->GetWorkingData();
}


//-----------------------------------------------------------------------------
mitk::Image* niftkBaseSegmentationView::GetReferenceImage()
{
  mitk::Image* result = this->GetReferenceImageFromToolManager();
  return result;
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentationView::IsNodeAReferenceImage(const mitk::DataNode::Pointer node)
{
  return mitk::IsNodeAGreyScaleImage(node);
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentationView::IsNodeASegmentationImage(const mitk::DataNode::Pointer node)
{
  return mitk::IsNodeABinaryImage(node);
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentationView::IsNodeAWorkingImage(const mitk::DataNode::Pointer node)
{
  return mitk::IsNodeABinaryImage(node);
}

mitk::ToolManager::DataVectorType niftkBaseSegmentationView::GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  // This default implementation just says Segmentation node == Working node, which subclasses could override.

  mitk::ToolManager::DataVectorType result(1);
  result[0] = node;
  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* niftkBaseSegmentationView::GetSegmentationNodeFromWorkingData(const mitk::DataNode::Pointer node)
{
  // This default implementation just says Segmentation node == Working node, which subclasses could override.

  mitk::DataNode::Pointer result = node;
  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* niftkBaseSegmentationView::CreateNewSegmentation(QColor &defaultColor)
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
        niftkMIDASNewSegmentationDialog* dialog = new niftkMIDASNewSegmentationDialog(defaultColor, this->GetParent() ); // needs a QWidget as parent, "this" is not QWidget
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
void niftkBaseSegmentationView::CreateConnections()
{
}


//-----------------------------------------------------------------------------
mitk::BaseRenderer* niftkBaseSegmentationView::GetFocusedRenderer()
{
  return QmitkBaseView::GetFocusedRenderer();
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentationView::SetEnableManualToolSelectionBox(bool enabled)
{
  m_BaseSegmentationViewControls->SetEnableManualToolSelectionBox(enabled);
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentationView::ApplyDisplayOptions(mitk::DataNode* node)
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
void niftkBaseSegmentationView::SetToolManagerSelection(const mitk::DataNode* referenceData, const mitk::ToolManager::DataVectorType workingDataNodes)
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
      m_BaseSegmentationViewControls->m_SegmentationSelectorWidget->m_NewSegmentationButton->setEnabled(true);
      m_BaseSegmentationViewControls->m_SegmentationSelectorWidget->m_SegmentationImageNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
    }
    else
    {
      mitk::DataNode::Pointer node = workingDataNodes[0];
      mitk::DataNode::Pointer segmentationImage = this->GetSegmentationNodeFromWorkingData(node);
      assert(segmentationImage);

      m_BaseSegmentationViewControls->m_SegmentationSelectorWidget->m_NewSegmentationButton->setEnabled(false);
      m_BaseSegmentationViewControls->m_SegmentationSelectorWidget->m_SegmentationImageNameLabel->setText(tr("<font color='black'>%1</font>").arg(segmentationImage->GetName().c_str()));
    }
  }
  else
  {
    m_BaseSegmentationViewControls->m_SegmentationSelectorWidget->m_NewSegmentationButton->setEnabled(false);
    m_BaseSegmentationViewControls->m_SegmentationSelectorWidget->m_SegmentationImageNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
  }
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentationView::GetSliceNumberFromSliceNavigationControllerAndReferenceImage()
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
MIDASOrientation niftkBaseSegmentationView::GetOrientationAsEnum()
{
  MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;
  const mitk::SliceNavigationController* sliceNavigationController = this->GetSliceNavigationController();
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
int niftkBaseSegmentationView::GetAxisFromReferenceImage(const MIDASOrientation& orientation)
{
  int axis = -1;
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  if (referenceImage.IsNotNull())
  {
    axis = niftk::GetThroughPlaneAxis(referenceImage, orientation);
  }
  return axis;
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentationView::GetReferenceImageAxialAxis()
{
  return this->GetAxisFromReferenceImage(MIDAS_ORIENTATION_AXIAL);
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentationView::GetReferenceImageCoronalAxis()
{
  return this->GetAxisFromReferenceImage(MIDAS_ORIENTATION_CORONAL);
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentationView::GetReferenceImageSagittalAxis()
{
  return this->GetAxisFromReferenceImage(MIDAS_ORIENTATION_SAGITTAL);
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentationView::GetViewAxis()
{
  int axisNumber = -1;
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  MIDASOrientation orientation = this->GetOrientationAsEnum();
  if (referenceImage.IsNotNull() && orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    axisNumber = niftk::GetThroughPlaneAxis(referenceImage, orientation);
  }
  return axisNumber;
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentationView::GetUpDirection()
{
  int upDirection = 0;
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager();
  MIDASOrientation orientation = this->GetOrientationAsEnum();
  if (referenceImage.IsNotNull() && orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    upDirection = niftk::GetUpDirection(referenceImage, orientation);
  }
  return upDirection;
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentationView::SetReferenceImageSelected()
{
  mitk::DataNode::Pointer referenceImageNode = this->GetReferenceNodeFromToolManager();
  if (referenceImageNode.IsNotNull())
  {
    this->SetCurrentSelection(referenceImageNode);
  }
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentationView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentationView::RetrievePreferenceValues()
{
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  assert( prefService );

  berry::IBerryPreferences::Pointer prefs
      = (prefService->GetSystemPreferences()->Node(this->GetPreferencesNodeName()))
        .Cast<berry::IBerryPreferences>();

  assert( prefs );

  QString defaultColorName = prefs->Get(niftkBaseSegmentationView::DEFAULT_COLOUR, "");
  m_DefaultSegmentationColor = QColor(defaultColorName);
  if (defaultColorName == "") // default values
  {
    m_DefaultSegmentationColor = QColor(0, 255, 0);
  }
}
