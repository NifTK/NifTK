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

#include <mitkStateEvent.h>
#include <mitkVtkResliceInterpolationProperty.h>

#include <QmitkRenderWindow.h>

#include <mitkDataStorageUtils.h>

#include <niftkIBaseView.h>

#include "Internal/niftkBaseSegmentorGUI.h"
#include "Internal/niftkNewSegmentationDialog.h"

namespace niftk
{

//-----------------------------------------------------------------------------
BaseSegmentorController::BaseSegmentorController(IBaseView* view)
  : BaseController(view),
    m_SegmentorGUI(nullptr),
    m_SelectedNode(nullptr),
    m_SelectedImage(nullptr),
    m_ActiveToolID(-1),
    m_CursorIsVisibleWhenToolsAreOff(true)
{
  // Create an own tool manager and connect it to the data storage straight away.
  m_ToolManager = mitk::ToolManager::New(view->GetDataStorage());
}


//-----------------------------------------------------------------------------
BaseSegmentorController::~BaseSegmentorController()
{
}


//-----------------------------------------------------------------------------
void BaseSegmentorController::SetupGUI(QWidget* parent)
{
  BaseController::SetupGUI(parent);

  m_SegmentorGUI = dynamic_cast<BaseSegmentorGUI*>(this->GetGUI());
  m_SegmentorGUI->SetToolManager(m_ToolManager);

  this->connect(m_SegmentorGUI, SIGNAL(NewSegmentationButtonClicked()), SLOT(OnNewSegmentationButtonClicked()));
  this->connect(m_SegmentorGUI, SIGNAL(ToolSelected(int)), SLOT(OnToolSelected(int)));
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
bool BaseSegmentorController::EventFilter(const mitk::StateEvent* stateEvent) const
{
  if (QmitkRenderWindow* renderWindow = this->GetView()->GetSelectedRenderWindow())
  {
    if (renderWindow->GetRenderer() == stateEvent->GetEvent()->GetSender())
    {
      return false;
    }
  }

  return true;
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
mitk::ToolManager::DataVectorType BaseSegmentorController::GetWorkingData()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  return toolManager->GetWorkingData();
}


//-----------------------------------------------------------------------------
mitk::Image* BaseSegmentorController::GetWorkingImage(int index)
{
  mitk::Image* result = nullptr;

  mitk::ToolManager::DataVectorType workingData = this->GetWorkingData();
  if (workingData.size() > 0 && index >= 0 && index < (int)workingData.size())
  {
    mitk::DataNode::Pointer node = workingData[index];

    if (node.IsNotNull())
    {
      mitk::Image* image = dynamic_cast<mitk::Image*>( node->GetData() );
      if (image)
      {
        result = image;
      }
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* BaseSegmentorController::GetReferenceNode()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  return toolManager->GetReferenceData(0);
}


//-----------------------------------------------------------------------------
mitk::Image* BaseSegmentorController::GetReferenceImage()
{
  mitk::Image* result = nullptr;

  mitk::DataNode* node = this->GetReferenceNode();
  if (node)
  {
    mitk::Image* image = dynamic_cast<mitk::Image*>( node->GetData() );
    if (image)
    {
      result = image;
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* BaseSegmentorController::FindReferenceNodeFromSegmentationNode(const mitk::DataNode::Pointer segmentationNode)
{
  mitk::DataNode* result = mitk::FindFirstParentImage(this->GetDataStorage(), segmentationNode, false);
  return result;
}


//-----------------------------------------------------------------------------
void BaseSegmentorController::SetReferenceImageSelected()
{
  mitk::DataNode::Pointer referenceImageNode = this->GetReferenceNode();
  if (referenceImageNode.IsNotNull())
  {
    this->GetView()->SetCurrentSelection(referenceImageNode);
  }
}


//-----------------------------------------------------------------------------
bool BaseSegmentorController::IsAReferenceImage(const mitk::DataNode::Pointer node)
{
  return mitk::IsNodeAGreyScaleImage(node);
}


//-----------------------------------------------------------------------------
bool BaseSegmentorController::IsASegmentationImage(const mitk::DataNode::Pointer node)
{
  return mitk::IsNodeABinaryImage(node);
}


//-----------------------------------------------------------------------------
bool BaseSegmentorController::IsAWorkingImage(const mitk::DataNode::Pointer node)
{
  return mitk::IsNodeABinaryImage(node);
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType BaseSegmentorController::GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  // This default implementation just says Segmentation node == Working node, which subclasses could override.

  mitk::ToolManager::DataVectorType result(1);
  result[0] = node;
  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* BaseSegmentorController::GetSegmentationNodeFromWorkingData(const mitk::DataNode::Pointer node)
{
  // This default implementation just says Segmentation node == Working node, which subclasses could override.

  mitk::DataNode::Pointer result = node;
  return result;
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
int BaseSegmentorController::GetReferenceImageSliceAxis()
{
  int referenceImageAxis = -1;
  mitk::Image::Pointer referenceImage = this->GetReferenceImage();
  ImageOrientation orientation = this->GetOrientation();
  if (referenceImage.IsNotNull() && orientation != IMAGE_ORIENTATION_UNKNOWN)
  {
    referenceImageAxis = GetThroughPlaneAxis(referenceImage, orientation);
  }
  return referenceImageAxis;
}


//-----------------------------------------------------------------------------
int BaseSegmentorController::GetReferenceImageSliceAxis(ImageOrientation orientation)
{
  int referenceImageSliceAxis = -1;
  mitk::Image* referenceImage = this->GetReferenceImage();
  if (referenceImage)
  {
    referenceImageSliceAxis = GetThroughPlaneAxis(referenceImage, orientation);
  }
  return referenceImageSliceAxis;
}


//-----------------------------------------------------------------------------
int BaseSegmentorController::GetReferenceImageSliceIndex()
{
  int referenceImageSliceIndex = -1;

  mitk::Image* referenceImage = this->GetReferenceImage();
  mitk::SliceNavigationController* snc = this->GetSliceNavigationController();

  if (referenceImage && snc)
  {
    const mitk::PlaneGeometry* planeGeometry = snc->GetCurrentPlaneGeometry();
    if (planeGeometry)
    {
      mitk::Point3D originInMm = planeGeometry->GetOrigin();
      mitk::Point3D originInVx;
      referenceImage->GetGeometry()->WorldToIndex(originInMm, originInVx);

      int viewAxis = this->GetReferenceImageSliceAxis();
      referenceImageSliceIndex = (int)(originInVx[viewAxis] + 0.5);
    }
  }
  return referenceImageSliceIndex;
}


//-----------------------------------------------------------------------------
int BaseSegmentorController::GetReferenceImageSliceUpDirection()
{
  int upDirection = 0;
  mitk::Image::Pointer referenceImage = this->GetReferenceImage();
  ImageOrientation orientation = this->GetOrientation();
  if (referenceImage.IsNotNull() && orientation != IMAGE_ORIENTATION_UNKNOWN)
  {
    upDirection = niftk::GetUpDirection(referenceImage, orientation);
  }
  return upDirection;
}


//-----------------------------------------------------------------------------
mitk::DataNode* BaseSegmentorController::CreateNewSegmentation()
{
  mitk::DataNode::Pointer emptySegmentation = NULL;

  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  // Assumption: If a reference image is selected in the data manager, then it MUST be registered with ToolManager, and hence this is the one we intend to segment.
  mitk::DataNode::Pointer referenceNode = this->GetReferenceNode();
  if (referenceNode.IsNotNull())
  {
    // Assumption: If a reference image is selected in the data manager, then it MUST be registered with ToolManager, and hence this is the one we intend to segment.
    mitk::Image::Pointer referenceImage = this->GetReferenceImage();
    if (referenceImage.IsNotNull())
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
void BaseSegmentorController::OnDataManagerSelectionChanged(const QList<mitk::DataNode::Pointer>& nodes)
{
  assert(m_SegmentorGUI);

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
    if (this->IsAReferenceImage(node))
    {
      referenceData = node;
    }

    // A segmentation image, is the final output, the one being segmented.
    if (this->IsASegmentationImage(node))
    {
      segmentedData = node;
    }
    else if (mitk::IsNodeABinaryImage(node) && this->CanStartSegmentationForBinaryNode(node))
    {
      segmentedData = node;
    }

    if (segmentedData.IsNotNull())
    {

      referenceData = this->FindReferenceNodeFromSegmentationNode(segmentedData);

      if (this->IsASegmentationImage(node))
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
mitk::DataNode::Pointer BaseSegmentorController::GetSelectedNode() const
{
  return m_SelectedNode;
}


//-----------------------------------------------------------------------------
void BaseSegmentorController::OnNewSegmentationButtonClicked()
{
}


//-----------------------------------------------------------------------------
void BaseSegmentorController::SetToolManagerSelection(const mitk::DataNode* referenceData, const mitk::ToolManager::DataVectorType workingDataNodes)
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
void BaseSegmentorController::OnViewGetsActivated()
{
  this->OnDataManagerSelectionChanged(this->GetDataManagerSelection());
}


//-----------------------------------------------------------------------------
void BaseSegmentorController::OnToolSelected(int toolID)
{
  /// Note: The view is not created when the GUI is set up, therefore
  /// we cannot initialise this variable at another place.
  static bool firstCall = true;
  if (firstCall)
  {
    firstCall = false;
    m_CursorIsVisibleWhenToolsAreOff = this->GetView()->IsActiveEditorCursorVisible();
  }

  if (toolID != -1)
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

  m_ActiveToolID = toolID;

  /// Set the focus back to the main window. This is needed so that the keyboard shortcuts
  /// (like 'a' and 'z' for changing slice) keep on working.
  if (QmitkRenderWindow* mainWindow = this->GetView()->GetSelectedRenderWindow())
  {
    mainWindow->setFocus();
  }
}

}
