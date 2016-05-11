/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkBaseSegmentorController_h
#define __niftkBaseSegmentorController_h

#include <niftkMIDASGuiExports.h>

#include <QColor>
#include <QList>

#include <mitkDataNode.h>
#include <mitkToolManager.h>

#include <niftkBaseController.h>
#include <niftkMIDASEventFilter.h>
#include <niftkImageOrientationUtils.h>

class QWidget;

class niftkBaseSegmentorGUI;
class niftkIBaseView;

/**
 * \class niftkBaseSegmentorController
 */
class NIFTKMIDASGUI_EXPORT niftkBaseSegmentorController : public niftk::BaseController, public niftk::MIDASEventFilter
{

  Q_OBJECT

public:

  niftkBaseSegmentorController(niftkIBaseView* view);

  virtual ~niftkBaseSegmentorController();

  /// \brief Sets up the GUI.
  /// This function has to be called from the CreateQtPartControl function of the view.
  virtual void SetupGUI(QWidget* parent) override;

  /// \brief Returns the segmentation tool manager used by the segmentor.
  mitk::ToolManager* GetToolManager() const;

  template <class ToolType>
  ToolType* GetToolByType();

  /// \brief Returns true if the event should be filtered, i.e. not processed,
  /// otherwise false.
  virtual bool EventFilter(const mitk::StateEvent* stateEvent) const override;

  /// \brief Returns true if the event should be filtered, i.e. not processed,
  /// otherwise false.
  virtual bool EventFilter(mitk::InteractionEvent* event) const override;

  /// \brief Default colour to be displayed in the new segmentation dialog box.
  const QColor& GetDefaultSegmentationColour() const;

  /// \brief Default colour to be displayed in the new segmentation dialog box.
  void SetDefaultSegmentationColour(const QColor& defaultSegmentationColour);

protected slots:

  /// \brief Called from niftkToolSelectorWidget when a tool changes.
  virtual void OnToolSelected(int toolID);

protected:

  /// \brief Gets a vector of the working data nodes registered with the tool manager.
  /// The data nodes normally hold image, but could be surfaces etc.
  /// Empty list is returned if this can't be found.
  mitk::ToolManager::DataVectorType GetWorkingData();

  /// \brief Gets a single binary image registered with the ToolManager.
  /// Returns nullptr if it can't be found or is not an image.
  mitk::Image* GetWorkingImageFromToolManager(int index);

  /// \brief Gets the reference node from the tool manager or nullptr if it can't be found.
  mitk::DataNode* GetReferenceNodeFromToolManager();

  /// \brief Gets the reference image from the tool manager, or nullptr if this doesn't yet exist or is not an image.
  mitk::Image* GetReferenceImageFromToolManager();

  /// \brief Gets the reference node that the segmentation node belongs to.
  /// Assumes that the reference (grey scale) node is always the direct parent of the
  /// segmentation (binary) node, so we simply search for a non binary parent.
  mitk::DataNode* GetReferenceNodeFromSegmentationNode(const mitk::DataNode::Pointer segmentationNode);

  /// \brief Gets the reference image registered with the tool manager.
  /// Assumes that a reference (grey scale) image is always registered with the tool manager.
  mitk::Image* GetReferenceImage();

  /// \brief Makes sure the reference image is the selected one
  void SetReferenceImageSelected();

  /// \brief Returns true if node represent an image that is non binary, and false otherwise.
  virtual bool IsNodeAReferenceImage(const mitk::DataNode::Pointer node);

  /// \brief Returns true if node represents an image that is binary, and false otherwise.
  virtual bool IsNodeASegmentationImage(const mitk::DataNode::Pointer node);

  /// \brief Returns true if node represents an image that is binary, and false otherwise.
  virtual bool IsNodeAWorkingImage(const mitk::DataNode::Pointer node);

  /// \brief Assumes that a Working Node == a Segmentation Node, so simply returns the input node.
  virtual mitk::ToolManager::DataVectorType GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer node);

  /// \brief Assumes that a Working Node == a Segmentation Node, so simply returns the input node.
  virtual mitk::DataNode* GetSegmentationNodeFromWorkingData(const mitk::DataNode::Pointer node);

  /// \brief Subclasses decide if they can restart the segmentation for a binary node.
  virtual bool CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node) = 0;

  /// \brief Decorates a DataNode according to the user preference settings, or requirements for binary images.
  virtual void ApplyDisplayOptions(mitk::DataNode* node);

  /// \brief Works out the slice number.
  int GetSliceNumberFromSliceNavigationControllerAndReferenceImage();

  /// \brief Looks up the ReferenceImage registered with ToolManager and returns the axis [0,1,2] that corresponds to the given orientation, or -1 if it can't be found.
  int GetAxisFromReferenceImage(niftk::ImageOrientation orientation);

  /// \brief Returns the reference image axial axis [0,1,2] or -1 if it can't be found.
  int GetReferenceImageAxialAxis();

  /// \brief Returns the reference image coronal axis [0,1,2] or -1 if it can't be found.
  int GetReferenceImageCoronalAxis();

  /// \brief Returns the reference image coronal axis [0,1,2] or -1 if it can't be found.
  int GetReferenceImageSagittalAxis();

  /// \brief Retrieves the currently active QmitkRenderWindow, and the reference image registered with the ToolManager, and returns the Image axis that the current view is looking along, or -1 if it can not be worked out.
  int GetViewAxis();

  /// \brief Returns the "Up" direction which is the anterior, superior or right direction depending on which orientation you are interested in.
  int GetUpDirection();

  /// \brief Creates from derived classes when the the user hits the "New segmentation", producing a dialog box,
  /// and on successful completion of the dialog box, will create a new segmentation image.
  ///
  /// \return mitk::DataNode* A new segmentation or <code>NULL</code> if the user cancels the dialog box.
  virtual mitk::DataNode* CreateNewSegmentation();

  /// \brief Gets the segmentor widget that holds the GUI components of the view.
  niftkBaseSegmentorGUI* GetSegmentorGUI() const;

  /// \brief Called when the selection changes in the data manager.
  /// \see QmitkAbstractView::OnSelectionChanged.
  virtual void OnDataManagerSelectionChanged(const QList<mitk::DataNode::Pointer>& nodes);

  /// \brief Returns the last selected node, whenever only a single node is selected. If you multi-select, this is not updated.
  mitk::DataNode::Pointer GetSelectedNode() const;

protected slots:

  /// \brief Called from niftkSegmentationSelectorWidget when the 'Start/restart segmentation' button is clicked.
  virtual void OnNewSegmentationButtonClicked();

private:

  /// \brief Propagate data manager selection to tool manager for manual segmentation.
  virtual void SetToolManagerSelection(const mitk::DataNode* referenceData, const mitk::ToolManager::DataVectorType workingDataNodes);

  mitk::ToolManager::Pointer m_ToolManager;

  niftkBaseSegmentorGUI* m_SegmentorGUI;

  /// \brief Keeps track of the last selected node, whenever only a single node is selected. If you multi-select, this is not updated.
  mitk::DataNode::Pointer m_SelectedNode;

  /// \brief Keeps track of the last selected image, whenever only a single node is selected. If you multi-select, this is not updated.
  mitk::Image::Pointer m_SelectedImage;

  /// \brief Default colour to be displayed in the new segmentation dialog box.
  QColor m_DefaultSegmentationColour;

  /// \brief The ID of the currently active tool or -1 if no tool is active.
  int m_ActiveToolID;

  /// \brief Stores the visibility state of the cursor in the main display before activating a tool.
  bool m_CursorIsVisibleWhenToolsAreOff;

friend class niftkBaseSegmentorView;

};


//-----------------------------------------------------------------------------
template <class ToolType>
ToolType* niftkBaseSegmentorController::GetToolByType()
{
  int toolId = m_ToolManager->GetToolIdByToolType<ToolType>();
  return dynamic_cast<ToolType*>(m_ToolManager->GetToolById(toolId));
}

#endif
