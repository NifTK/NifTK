/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseSegmentorController.h"

#include <QMessageBox>

#include <mitkUndoController.h>
#include <mitkVtkResliceInterpolationProperty.h>

#include <QmitkRenderWindow.h>

#include <niftkDataStorageUtils.h>
#include <niftkIBaseView.h>

#include "Internal/niftkBaseSegmentorGUI.h"
#include "Internal/niftkNewSegmentationDialog.h"

namespace niftk
{

//-----------------------------------------------------------------------------
BaseSegmentorController::BaseSegmentorController(IBaseView* view)
  : BaseController(view),
    m_SegmentorGUI(nullptr),
    m_ActiveToolID(-1),
    m_CursorIsVisibleWhenToolsAreOff(true)
{
  // Create an own tool manager and connect it to the data storage straight away.
  m_ToolManager = mitk::ToolManager::New(view->GetDataStorage());
}


//-----------------------------------------------------------------------------
BaseSegmentorController::~BaseSegmentorController()
{
  m_ToolManager->ActiveToolChanged -= mitk::MessageDelegate<BaseSegmentorController>(this, &BaseSegmentorController::OnActiveToolChanged);
  m_ToolManager->ReferenceDataChanged -= mitk::MessageDelegate<BaseSegmentorController>(this, &BaseSegmentorController::OnReferenceNodesChanged);
  m_ToolManager->WorkingDataChanged -= mitk::MessageDelegate<BaseSegmentorController>(this, &BaseSegmentorController::OnWorkingNodesChanged);
}


//-----------------------------------------------------------------------------
void BaseSegmentorController::SetupGUI(QWidget* parent)
{
  BaseController::SetupGUI(parent);

  m_SegmentorGUI = dynamic_cast<BaseSegmentorGUI*>(this->GetGUI());
  m_SegmentorGUI->SetToolManager(m_ToolManager);

  m_ToolManager->ActiveToolChanged += mitk::MessageDelegate<BaseSegmentorController>(this, &BaseSegmentorController::OnActiveToolChanged);
  m_ToolManager->ReferenceDataChanged += mitk::MessageDelegate<BaseSegmentorController>(this, &BaseSegmentorController::OnReferenceNodesChanged);
  m_ToolManager->WorkingDataChanged += mitk::MessageDelegate<BaseSegmentorController>(this, &BaseSegmentorController::OnWorkingNodesChanged);

  this->connect(m_SegmentorGUI, SIGNAL(NewSegmentationButtonClicked()), SLOT(OnNewSegmentationButtonClicked()));
}


//-----------------------------------------------------------------------------
BaseSegmentorGUI* BaseSegmentorController::GetSegmentorGUI() const
{
  return m_SegmentorGUI;
}


//-----------------------------------------------------------------------------
const QColor& BaseSegmentorController::GetDefaultSegmentationColour() const
{
  return m_DefaultSegmentationColour;
}


//-----------------------------------------------------------------------------
void BaseSegmentorController::SetDefaultSegmentationColour(const QColor& defaultSegmentationColour)
{
  m_DefaultSegmentationColour = defaultSegmentationColour;
}


//-----------------------------------------------------------------------------
bool BaseSegmentorController::EventFilter(mitk::InteractionEvent* event) const
{
  if (QmitkRenderWindow* renderWindow = this->GetView()->GetSelectedRenderWindow())
  {
    if (renderWindow->GetRenderer() == event->GetSender())
    {
      return false;
    }
  }

  return true;
}


//-----------------------------------------------------------------------------
mitk::ToolManager* BaseSegmentorController::GetToolManager() const
{
  return m_ToolManager;
}


//-----------------------------------------------------------------------------
std::vector<mitk::DataNode*> BaseSegmentorController::GetReferenceNodes()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  return toolManager->GetReferenceData();
}


//-----------------------------------------------------------------------------
mitk::DataNode* BaseSegmentorController::GetReferenceNode(int index)
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  return toolManager->GetReferenceData(index);
}


//-----------------------------------------------------------------------------
mitk::Image* BaseSegmentorController::GetReferenceImage(int index)
{
  if (auto node = this->GetReferenceNode(index))
  {
    return dynamic_cast<mitk::Image*>(node->GetData());
  }

  return nullptr;
}


//-----------------------------------------------------------------------------
std::vector<mitk::DataNode*> BaseSegmentorController::GetWorkingNodes()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  return toolManager->GetWorkingData();
}


//-----------------------------------------------------------------------------
mitk::DataNode* BaseSegmentorController::GetWorkingNode(int index)
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  return toolManager->GetWorkingData(index);
}


//-----------------------------------------------------------------------------
mitk::Image* BaseSegmentorController::GetWorkingImage(int index)
{
  if (auto node = this->GetWorkingNode(index))
  {
    return dynamic_cast<mitk::Image*>(node->GetData());
  }

  return nullptr;
}


//-----------------------------------------------------------------------------
std::vector<mitk::DataNode*> BaseSegmentorController::GetWorkingNodesFrom(mitk::DataNode* segmentationNode)
{
  /// This default implementation just says Segmentation node == Working node, which subclasses could override.
  /// Every derived class should store the segmentation node in the first (0th) element of the vector, though.

  std::vector<mitk::DataNode*> result(1);
  result[0] = segmentationNode;
  return result;
}


//-----------------------------------------------------------------------------
bool BaseSegmentorController::CanStartSegmentationFrom(const mitk::DataNode* node)
{
  bool canRestart = false;

  if (node && niftk::IsNodeABinaryImage(node))
  {
    mitk::DataNode* parent = niftk::FindFirstParentImage(this->GetDataStorage(), node, false);
    if (parent)
    {
      if (niftk::IsNodeANonBinaryImage(parent))
      {
        canRestart = true;
      }
    }
  }

  return canRestart;
}


//-----------------------------------------------------------------------------
void BaseSegmentorController::ApplyDisplayOptions(mitk::DataNode* node)
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
mitk::DataNode::Pointer BaseSegmentorController::CreateNewSegmentation()
{
  mitk::DataNode::Pointer emptySegmentation;

  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  // Assumption: If a reference image is selected in the data manager, then it MUST be registered with ToolManager, and hence this is the one we intend to segment.
  mitk::DataNode* referenceNode = this->GetReferenceNode();
  if (referenceNode)
  {
    // Assumption: If a reference image is selected in the data manager, then it MUST be registered with ToolManager, and hence this is the one we intend to segment.
    const mitk::Image* referenceImage = this->GetReferenceImage();
    if (referenceImage)
    {
      if (referenceImage->GetDimension() > 2)
      {
        NewSegmentationDialog* dialog = new NewSegmentationDialog(m_DefaultSegmentationColour, m_SegmentorGUI->GetParent());
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

            if (emptySegmentation)
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
  return emptySegmentation;
}


//-----------------------------------------------------------------------------
bool BaseSegmentorController::HasWorkingNodes()
{
  return !this->GetWorkingNodes().empty();
}


//-----------------------------------------------------------------------------
void BaseSegmentorController::OnDataManagerSelectionChanged(const QList<mitk::DataNode::Pointer>& selectedNodes)
{
  mitk::DataNode* referenceNode = nullptr;
  std::vector<mitk::DataNode*> workingNodes;

  // This plugin only works if you single select, anything else is invalid (for now).
  if (selectedNodes.size() == 1)
  {
    // MAJOR ASSUMPTION: To get a segmentation plugin (i.e. all derived classes) to work, you select the segmentation node.
    // From this segmentation node, you can work out the reference data (always the parent).
    // In addition, you can work out any intermediate working images (either that image, or children).
    // MAJOR ASSUMPTION: Intermediate working images will be hidden, and hence not clickable.

    mitk::DataNode* selectedNode = selectedNodes[0];
    // Rely on subclasses deciding if the node is something we are interested in.
    if (niftk::IsNodeANonBinaryImage(selectedNode) && this->HasSameGeometryAsViewer(selectedNode))
    {
      referenceNode = selectedNode;
    }

    // A segmentation image, is the final output, the one being segmented.
    if (niftk::IsNodeABinaryImage(selectedNode))
    {
      /// This finds the first not binary parent.
      mitk::DataNode* potentialReferenceNode = niftk::FindFirstParentImage(this->GetDataStorage(), selectedNode, false);

      if (potentialReferenceNode && this->HasSameGeometryAsViewer(potentialReferenceNode))
      {
        std::vector<mitk::DataNode*> potentialWorkingNodes = this->GetWorkingNodesFrom(selectedNode);
        if (!potentialWorkingNodes.empty()
            || this->CanStartSegmentationFrom(selectedNode))
        {
          referenceNode = potentialReferenceNode;
          workingNodes = potentialWorkingNodes;
        }
      }
    }
  }

  mitk::UndoController::GetCurrentUndoModel()->Clear();

  // Tell the tool manager the images for reference and working purposes.
  this->SetToolManagerSelection(referenceNode, workingNodes);
}


//-----------------------------------------------------------------------------
bool BaseSegmentorController::HasSameGeometryAsViewer(mitk::DataNode* node)
{
  assert(node);

  auto data = node->GetData();
  auto snc = this->GetSliceNavigationController();

  return data && snc && data->GetTimeGeometry() == snc->GetInputWorldTimeGeometry();
}


//-----------------------------------------------------------------------------
void BaseSegmentorController::SetToolManagerSelection(mitk::DataNode* referenceNode, const std::vector<mitk::DataNode*>& workingNodes)
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  if (workingNodes.size() == 0 ||
      ( toolManager->GetWorkingData().size() > 0 &&
        workingNodes.size() > 0 &&
        toolManager->GetWorkingData(0) != workingNodes[0] ))
  {
    toolManager->ActivateTool(-1);
  }

  toolManager->SetReferenceData(referenceNode);
  toolManager->SetWorkingData(workingNodes);
}


//-----------------------------------------------------------------------------
void BaseSegmentorController::OnViewGetsActivated()
{
  this->OnDataManagerSelectionChanged(this->GetDataManagerSelection());
}


//-----------------------------------------------------------------------------
void BaseSegmentorController::OnActiveToolChanged()
{
  int activeToolID = m_ToolManager->GetActiveToolID();

  /// Note: The view is not created when the GUI is set up, therefore
  /// we cannot initialise this variable at another place.
  static bool firstCall = true;
  if (firstCall)
  {
    firstCall = false;
    m_CursorIsVisibleWhenToolsAreOff = this->GetView()->IsActiveEditorCursorVisible();
  }

  if (activeToolID != -1)
  {
    bool cursorWasVisible = this->GetView()->IsActiveEditorCursorVisible();
    if (cursorWasVisible)
    {
      this->GetView()->SetActiveEditorCursorVisible(false);
    }

    if (m_ActiveToolID == -1)
    {
      m_CursorIsVisibleWhenToolsAreOff = cursorWasVisible;
    }
  }
  else
  {
    this->GetView()->SetActiveEditorCursorVisible(m_CursorIsVisibleWhenToolsAreOff);
  }

  m_ActiveToolID = activeToolID;

  /// Set the focus back to the main window. This is needed so that the keyboard shortcuts
  /// (like 'a' and 'z' for changing slice) keep on working.
  if (QmitkRenderWindow* mainWindow = this->GetView()->GetSelectedRenderWindow())
  {
    mainWindow->setFocus();
  }
}


//-----------------------------------------------------------------------------
void BaseSegmentorController::OnReferenceNodesChanged()
{
  MITK_INFO << "BaseSegmentorController::OnReferenceNodesChanged()";
}


//-----------------------------------------------------------------------------
void BaseSegmentorController::OnWorkingNodesChanged()
{
  MITK_INFO << "BaseSegmentorController::OnWorkingNodesChanged()";
}

}
