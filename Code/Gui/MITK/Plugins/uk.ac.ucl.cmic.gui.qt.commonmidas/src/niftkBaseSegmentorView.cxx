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
  : m_ActiveToolID(-1),
    m_MainWindowCursorVisibleWithToolsOff(true)
{
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorView::~niftkBaseSegmentorView()
{
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::Activated()
{
  QmitkBaseView::Activated();

  assert(m_SegmentorController);
  m_SegmentorController->OnDataManagerSelectionChanged(this->GetDataManagerSelection());
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorView::niftkBaseSegmentorView(const niftkBaseSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::CreateQtPartControl(QWidget* parent)
{
  this->SetParent(parent);

  // Retrieving preferences done in another method so we can call it on startup, and when prefs change.
  this->RetrievePreferenceValues();

  m_SegmentorController = this->CreateSegmentorController();
  m_SegmentorController->SetupSegmentorGUI(parent);
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
void niftkBaseSegmentorView::OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer>& nodes)
{
  Q_UNUSED(part);
  assert(m_SegmentorController);
  m_SegmentorController->OnDataManagerSelectionChanged(nodes);
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
  return m_SegmentorController->CreateNewSegmentation(defaultColour);
}


//-----------------------------------------------------------------------------
mitk::BaseRenderer* niftkBaseSegmentorView::GetFocusedRenderer()
{
  return QmitkBaseView::GetFocusedRenderer();
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::ApplyDisplayOptions(mitk::DataNode* node)
{
  m_SegmentorController->ApplyDisplayOptions(node);
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorView::GetSliceNumberFromSliceNavigationControllerAndReferenceImage()
{
  return m_SegmentorController->GetSliceNumberFromSliceNavigationControllerAndReferenceImage();
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
  return m_SegmentorController->GetSelectedNode();
}


//-----------------------------------------------------------------------------
const QColor& niftkBaseSegmentorView::GetDefaultSegmentationColor() const
{
  return m_DefaultSegmentationColor;
}
