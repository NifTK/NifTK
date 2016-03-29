/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseSegmentorView.h"

#include <berryPlatform.h>

#include <QMessageBox>
#include <mitkILinkedRenderWindowPart.h>
#include <mitkImageAccessByItk.h>
#include <mitkDataNodeObject.h>
#include <mitkProperties.h>
#include <mitkColorProperty.h>
#include <mitkRenderingManager.h>
#include <mitkBaseRenderer.h>
#include <mitkSegTool2D.h>
#include <mitkToolManager.h>
#include <mitkGlobalInteraction.h>
#include <mitkStateMachine.h>
#include <mitkDataStorageUtils.h>
#include <mitkColorProperty.h>
#include <mitkProperties.h>
#include <QmitkRenderWindow.h>

#include <NifTKConfigure.h>
#include <niftkBaseSegmentorController.h>
#include <niftkBaseSegmentorGUI.h>
#include <niftkNewSegmentationDialog.h>
#include <niftkMIDASTool.h>
#include <niftkMIDASDrawTool.h>
#include <niftkMIDASPolyTool.h>
#include <niftkMIDASSeedTool.h>
#include <niftkMIDASOrientationUtils.h>

const QString niftkBaseSegmentorView::DEFAULT_COLOUR("midas editor default colour");
const QString niftkBaseSegmentorView::DEFAULT_COLOUR_STYLE_SHEET("midas editor default colour style sheet");


//-----------------------------------------------------------------------------
niftkBaseSegmentorView::niftkBaseSegmentorView()
  : m_SelectedNode(nullptr),
    m_SelectedImage(nullptr),
    m_ActiveToolID(-1),
    m_MainWindowCursorVisibleWithToolsOff(true),
    m_SegmentorGUI(nullptr)
{
  m_SelectedNode = NULL;
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorView::~niftkBaseSegmentorView()
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

  delete m_SegmentorGUI;
  m_SegmentorGUI = nullptr;
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentorView::EventFilter(const mitk::StateEvent* stateEvent) const
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
bool niftkBaseSegmentorView::EventFilter(mitk::InteractionEvent* event) const
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
void niftkBaseSegmentorView::Activated()
{
  QmitkBaseView::Activated();

  berry::IWorkbenchPart::Pointer nullPart;
  this->OnSelectionChanged(nullPart, this->GetDataManagerSelection());
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorView::niftkBaseSegmentorView(const niftkBaseSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::CreateQtPartControl(QWidget *parent)
{
  this->SetParent(parent);

  m_SegmentorController = this->CreateSegmentorController();

  mitk::ToolManager::Pointer toolManager = m_SegmentorController->GetToolManager();

  mitk::ToolManager::ToolVectorTypeConst tools = toolManager->GetTools();
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

  m_SegmentorGUI = this->CreateSegmentorGUI(parent);
  m_SegmentorGUI->SetToolManager(toolManager.GetPointer());

  this->connect(m_SegmentorGUI, SIGNAL(NewSegmentationButtonClicked()), SLOT(OnNewSegmentationButtonClicked()));
  this->connect(m_SegmentorGUI, SIGNAL(ToolSelected(int)), SLOT(OnToolSelected(int)));
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorGUI* niftkBaseSegmentorView::CreateSegmentorGUI(QWidget* parent)
{
  return m_SegmentorController->CreateSegmentorGUI(parent);
}


//-----------------------------------------------------------------------------
mitk::ToolManager* niftkBaseSegmentorView::GetToolManager()
{
  assert(m_SegmentorController);

  return m_SegmentorController->GetToolManager();
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::OnToolSelected(int toolID)
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
void niftkBaseSegmentorView::OnNewSegmentationButtonClicked()
{
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::OnSelectionChanged(berry::IWorkbenchPart::Pointer /*part*/, const QList<mitk::DataNode::Pointer> &nodes)
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
    if (referenceData.IsNotNull())
    {
      m_SegmentorGUI->SelectReferenceImage(QString::fromStdString(referenceData->GetName()));
    }
    else
    {
      m_SegmentorGUI->SelectReferenceImage();
    }

    // Tell the tool manager the images for reference and working purposes.
    this->SetToolManagerSelection(referenceData, workingDataNodes);

  }

  // Adjust widgets according to whether we have a valid selection.
  m_SegmentorGUI->EnableSegmentationWidgets(valid);
}


//-----------------------------------------------------------------------------
mitk::Image* niftkBaseSegmentorView::GetWorkingImageFromToolManager(int index)
{
  return m_SegmentorController->GetWorkingImageFromToolManager(index);
}


//-----------------------------------------------------------------------------
mitk::DataNode* niftkBaseSegmentorView::GetReferenceNodeFromToolManager()
{
  return m_SegmentorController->GetReferenceNodeFromToolManager();
}


//-----------------------------------------------------------------------------
mitk::Image* niftkBaseSegmentorView::GetReferenceImageFromToolManager()
{
  return m_SegmentorController->GetReferenceImageFromToolManager();
}


//-----------------------------------------------------------------------------
mitk::DataNode* niftkBaseSegmentorView::GetReferenceNodeFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  return m_SegmentorController->GetReferenceNodeFromSegmentationNode(node);
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType niftkBaseSegmentorView::GetWorkingData()
{
  return m_SegmentorController->GetWorkingData();
}


//-----------------------------------------------------------------------------
mitk::Image* niftkBaseSegmentorView::GetReferenceImage()
{
  return m_SegmentorController->GetReferenceImage();
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentorView::IsNodeAReferenceImage(const mitk::DataNode::Pointer node)
{
  return m_SegmentorController->IsNodeAReferenceImage(node);
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentorView::IsNodeASegmentationImage(const mitk::DataNode::Pointer node)
{
  return m_SegmentorController->IsNodeASegmentationImage(node);
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentorView::IsNodeAWorkingImage(const mitk::DataNode::Pointer node)
{
  return m_SegmentorController->IsNodeAWorkingImage(node);
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType niftkBaseSegmentorView::GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer segmentationNode)
{
  return m_SegmentorController->GetWorkingDataFromSegmentationNode(segmentationNode);
}


//-----------------------------------------------------------------------------
mitk::DataNode* niftkBaseSegmentorView::GetSegmentationNodeFromWorkingData(const mitk::DataNode::Pointer node)
{
  return m_SegmentorController->GetSegmentationNodeFromWorkingData(node);
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentorView::CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node)
{
  return m_SegmentorController->CanStartSegmentationForBinaryNode(node);
}


//-----------------------------------------------------------------------------
mitk::DataNode* niftkBaseSegmentorView::CreateNewSegmentation(const QColor& defaultColour)
{
  return m_SegmentorController->CreateNewSegmentation(this->GetParent(), defaultColour);
}


//-----------------------------------------------------------------------------
mitk::BaseRenderer* niftkBaseSegmentorView::GetFocusedRenderer()
{
  return QmitkBaseView::GetFocusedRenderer();
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::SetToolSelectorEnabled(bool enabled)
{
  m_SegmentorGUI->SetToolSelectorEnabled(enabled);
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::ApplyDisplayOptions(mitk::DataNode* node)
{
  m_SegmentorController->ApplyDisplayOptions(node);
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::SetToolManagerSelection(const mitk::DataNode* referenceData, const mitk::ToolManager::DataVectorType workingDataNodes)
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

  if (referenceData && !workingDataNodes.empty())
  {
    mitk::DataNode::Pointer node = workingDataNodes[0];
    mitk::DataNode::Pointer segmentationImage = this->GetSegmentationNodeFromWorkingData(node);
    assert(segmentationImage);
    m_SegmentorGUI->SelectSegmentationImage(QString::fromStdString(segmentationImage->GetName()));
  }
  else
  {
    m_SegmentorGUI->SelectSegmentationImage();
  }
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorView::GetSliceNumberFromSliceNavigationControllerAndReferenceImage()
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
MIDASOrientation niftkBaseSegmentorView::GetOrientationAsEnum()
{
  return m_SegmentorController->GetOrientationAsEnum();
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorView::GetAxisFromReferenceImage(const MIDASOrientation& orientation)
{
  return m_SegmentorController->GetAxisFromReferenceImage(orientation);
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorView::GetReferenceImageAxialAxis()
{
  return m_SegmentorController->GetReferenceImageAxialAxis();
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorView::GetReferenceImageCoronalAxis()
{
  return m_SegmentorController->GetReferenceImageCoronalAxis();
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorView::GetReferenceImageSagittalAxis()
{
  return m_SegmentorController->GetReferenceImageSagittalAxis();
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorView::GetViewAxis()
{
  return m_SegmentorController->GetViewAxis();
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorView::GetUpDirection()
{
  return m_SegmentorController->GetUpDirection();
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::SetReferenceImageSelected()
{
  mitk::DataNode::Pointer referenceImageNode = this->GetReferenceNodeFromToolManager();
  if (referenceImageNode.IsNotNull())
  {
    this->SetCurrentSelection(referenceImageNode);
  }
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::RetrievePreferenceValues()
{
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  assert( prefService );

  berry::IBerryPreferences::Pointer prefs
      = (prefService->GetSystemPreferences()->Node(this->GetPreferencesNodeName()))
        .Cast<berry::IBerryPreferences>();

  assert( prefs );

  QString defaultColorName = prefs->Get(niftkBaseSegmentorView::DEFAULT_COLOUR, "");
  m_DefaultSegmentationColor = QColor(defaultColorName);
  if (defaultColorName == "") // default values
  {
    m_DefaultSegmentationColor = QColor(0, 255, 0);
  }
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer niftkBaseSegmentorView::GetSelectedNode() const
{
  return m_SelectedNode;
}


//-----------------------------------------------------------------------------
const QColor& niftkBaseSegmentorView::GetDefaultSegmentationColor() const
{
  return m_DefaultSegmentationColor;
}
