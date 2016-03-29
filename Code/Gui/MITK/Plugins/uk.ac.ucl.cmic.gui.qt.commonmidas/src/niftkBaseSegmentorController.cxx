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

#include <mitkDataStorageUtils.h>
#include <mitkVtkResliceInterpolationProperty.h>

#include <niftkNewSegmentationDialog.h>

#include "niftkBaseSegmentorGUI.h"
#include "niftkBaseSegmentorView.h"

//-----------------------------------------------------------------------------
niftkBaseSegmentorController::niftkBaseSegmentorController(niftkBaseSegmentorView* segmentorView)
  : m_SegmentorGUI(nullptr),
    m_SegmentorView(segmentorView),
    m_SelectedNode(nullptr),
    m_SelectedImage(nullptr)
{
  // Create an own tool manager and connect it to the data storage straight away.
  m_ToolManager = mitk::ToolManager::New(segmentorView->GetDataStorage());
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorController::~niftkBaseSegmentorController()
{
}


//-----------------------------------------------------------------------------
mitk::DataStorage* niftkBaseSegmentorController::GetDataStorage() const
{
  return m_SegmentorView->GetDataStorage();
}


//-----------------------------------------------------------------------------
mitk::SliceNavigationController* niftkBaseSegmentorController::GetSliceNavigationController() const
{
  return m_SegmentorView->GetSliceNavigationController();
}


//-----------------------------------------------------------------------------
mitk::ToolManager* niftkBaseSegmentorController::GetToolManager() const
{
  return m_ToolManager;
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType niftkBaseSegmentorController::GetWorkingData()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  return toolManager->GetWorkingData();
}


//-----------------------------------------------------------------------------
mitk::Image* niftkBaseSegmentorController::GetWorkingImageFromToolManager(int index)
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
mitk::DataNode* niftkBaseSegmentorController::GetReferenceNodeFromToolManager()
{
  mitk::ToolManager* toolManager = this->GetToolManager();
  assert(toolManager);

  return toolManager->GetReferenceData(0);
}


//-----------------------------------------------------------------------------
mitk::Image* niftkBaseSegmentorController::GetReferenceImageFromToolManager()
{
  mitk::Image* result = nullptr;

  mitk::DataNode* node = this->GetReferenceNodeFromToolManager();
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
mitk::DataNode* niftkBaseSegmentorController::GetReferenceNodeFromSegmentationNode(const mitk::DataNode::Pointer segmentationNode)
{
  mitk::DataNode* result = mitk::FindFirstParentImage(this->GetDataStorage(), segmentationNode, false);
  return result;
}


//-----------------------------------------------------------------------------
mitk::Image* niftkBaseSegmentorController::GetReferenceImage()
{
  mitk::Image* result = this->GetReferenceImageFromToolManager();
  return result;
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentorController::IsNodeAReferenceImage(const mitk::DataNode::Pointer node)
{
  return mitk::IsNodeAGreyScaleImage(node);
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentorController::IsNodeASegmentationImage(const mitk::DataNode::Pointer node)
{
  return mitk::IsNodeABinaryImage(node);
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentorController::IsNodeAWorkingImage(const mitk::DataNode::Pointer node)
{
  return mitk::IsNodeABinaryImage(node);
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType niftkBaseSegmentorController::GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer node)
{
  // This default implementation just says Segmentation node == Working node, which subclasses could override.

  mitk::ToolManager::DataVectorType result(1);
  result[0] = node;
  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* niftkBaseSegmentorController::GetSegmentationNodeFromWorkingData(const mitk::DataNode::Pointer node)
{
  // This default implementation just says Segmentation node == Working node, which subclasses could override.

  mitk::DataNode::Pointer result = node;
  return result;
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorController::ApplyDisplayOptions(mitk::DataNode* node)
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
int niftkBaseSegmentorController::GetSliceNumberFromSliceNavigationControllerAndReferenceImage()
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
MIDASOrientation niftkBaseSegmentorController::GetOrientationAsEnum()
{
  MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;
  const mitk::SliceNavigationController* sliceNavigationController = m_SegmentorView->GetSliceNavigationController();
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
int niftkBaseSegmentorController::GetAxisFromReferenceImage(const MIDASOrientation& orientation)
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
int niftkBaseSegmentorController::GetReferenceImageAxialAxis()
{
  return this->GetAxisFromReferenceImage(MIDAS_ORIENTATION_AXIAL);
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorController::GetReferenceImageCoronalAxis()
{
  return this->GetAxisFromReferenceImage(MIDAS_ORIENTATION_CORONAL);
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorController::GetReferenceImageSagittalAxis()
{
  return this->GetAxisFromReferenceImage(MIDAS_ORIENTATION_SAGITTAL);
}


//-----------------------------------------------------------------------------
int niftkBaseSegmentorController::GetViewAxis()
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
int niftkBaseSegmentorController::GetUpDirection()
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
mitk::DataNode* niftkBaseSegmentorController::CreateNewSegmentation(QWidget* parent, const QColor& defaultColor)
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
        niftkNewSegmentationDialog* dialog = new niftkNewSegmentationDialog(defaultColor, parent); // needs a QWidget as parent, "this" is not QWidget
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
niftkBaseSegmentorGUI* niftkBaseSegmentorController::GetSegmentorGUI() const
{
  return m_SegmentorGUI;
}




//-----------------------------------------------------------------------------
void niftkBaseSegmentorController::OnDataManagerSelectionChanged(const QList<mitk::DataNode::Pointer>& nodes)
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
mitk::DataNode::Pointer niftkBaseSegmentorController::GetSelectedNode() const
{
  return m_SelectedNode;
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorController::SetToolManagerSelection(const mitk::DataNode* referenceData, const mitk::ToolManager::DataVectorType workingDataNodes)
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
