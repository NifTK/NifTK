/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkBaseSegmentorController_h
#define niftkBaseSegmentorController_h

#include <niftkMIDASGuiExports.h>

#include <QColor>
#include <QList>

#include <mitkDataNode.h>
#include <mitkToolManager.h>

#include <niftkBaseController.h>
#include <niftkStateMachineEventFilter.h>
#include <niftkImageOrientationUtils.h>

class QWidget;

namespace niftk
{

class BaseSegmentorGUI;


/// \class BaseSegmentorController
class NIFTKMIDASGUI_EXPORT BaseSegmentorController : public BaseController, public StateMachineEventFilter
{

  Q_OBJECT

public:

  BaseSegmentorController(IBaseView* view);

  virtual ~BaseSegmentorController();

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

  /// \brief Called when the BlueBerry view that hosts the GUI for this controller gets activated.
  virtual void OnViewGetsActivated() override;

protected:

  /// \brief Called from niftkToolSelectorWidget when a tool changes.
  virtual void OnActiveToolChanged();

  /// \brief Called when the reference data nodes have changed.
  virtual void OnReferenceNodesChanged();

  /// \brief Called when the working data nodes have changed.
  virtual void OnWorkingNodesChanged();

  /// \brief Gets the vector of the reference data nodes registered with the tool manager.
  /// The data nodes normally holds one grey scale image.
  /// Empty list is returned if this can't be found.
  std::vector<mitk::DataNode*> GetReferenceNodes() const;

  /// \brief Gets the reference node with the given index from the tool manager or nullptr if it can't be found.
  /// Normally, there is only one reference image with 0 index.
  mitk::DataNode* GetReferenceNode(int index = 0) const;

  /// \brief Gets the reference image with the given index from the tool manager, or nullptr if this doesn't yet exist or is not an image.
  /// Normally, there is only one reference image with 0 index.
  const mitk::Image* GetReferenceImage(int index = 0) const;

  /// \brief Gets the vector of the working data nodes registered with the tool manager.
  /// The data nodes normally hold image, but could be surfaces, point sets etc.
  /// Empty list is returned if this can't be found.
  std::vector<mitk::DataNode*> GetWorkingNodes() const;

  /// \brief Gets the working data node with the given index from the tool manager or nullptr if it can't be found.
  /// The data node of the segmented image has the 0 index.
  mitk::DataNode* GetWorkingNode(int index = 0) const;

  /// \brief Gets the image in working data node with the given index registered with the ToolManager.
  /// The segmented image has the 0 index.
  /// Returns nullptr if it can't be found or is not an image.
  mitk::Image* GetWorkingImage(int index = 0) const;

  /// \brief Assumes that a Working Node == a Segmentation Node, so simply returns the input node.
  virtual std::vector<mitk::DataNode*> GetWorkingNodesFrom(mitk::DataNode* segmentationNode);

  /// \brief Tells if this segmentor supports segmenting the given image.
  /// This implementation allows only grey scale images.
  virtual bool IsNodeAValidReferenceImage(const mitk::DataNode* node);

  /// \brief Tells if this segmentor supports editing the given segmentation.
  /// This implementation allows any binary image that has a parent that is a valid reference image.
  virtual bool IsNodeAValidSegmentationImage(const mitk::DataNode* node);

  /// \brief Decorates a DataNode according to the user preference settings, or requirements for binary images.
  virtual void ApplyDisplayOptions(mitk::DataNode* node);

  /// \brief Creates from derived classes when the the user hits the "New segmentation", producing a dialog box,
  /// and on successful completion of the dialog box, will create a new segmentation image.
  ///
  /// \return mitk::DataNode* A new segmentation or <code>NULL</code> if the user cancels the dialog box.
  virtual mitk::DataNode::Pointer CreateNewSegmentation();

  /// \brief Gets the segmentor widget that holds the GUI components of the view.
  BaseSegmentorGUI* GetSegmentorGUI() const;

  /// \brief Utility method to check that we have initialised all the working data nodes such as contours, region growing images etc.
  bool HasWorkingNodes() const;

  /// \brief Checks if the time geometry of the node data is the same as that of the focused renderer.
  bool HasSameGeometryAsViewer(mitk::DataNode* node);

  /// \brief Called when the selection changes in the data manager.
  /// \see QmitkAbstractView::OnSelectionChanged.
  virtual void OnDataManagerSelectionChanged(const QList<mitk::DataNode::Pointer>& nodes);

protected slots:

  /// \brief Called from niftkSegmentationSelectorWidget when the 'Start/restart segmentation' button is clicked.
  virtual void OnNewSegmentationButtonClicked() = 0;

private:

  /// \brief Propagate data manager selection to tool manager for manual segmentation.
  virtual void SetToolManagerSelection(mitk::DataNode* referenceData, const std::vector<mitk::DataNode*>& workingNodes);

  mitk::ToolManager::Pointer m_ToolManager;

  BaseSegmentorGUI* m_SegmentorGUI;

  /// \brief Default colour to be displayed in the new segmentation dialog box.
  QColor m_DefaultSegmentationColour;

  /// \brief The ID of the currently active tool or -1 if no tool is active.
  int m_ActiveToolID;

  /// \brief Stores the visibility state of the cursor in the main display before activating a tool.
  bool m_CursorIsVisibleWhenToolsAreOff;

friend class BaseSegmentorView;

};


//-----------------------------------------------------------------------------
template <class ToolType>
ToolType* BaseSegmentorController::GetToolByType()
{
  int toolId = m_ToolManager->GetToolIdByToolType<ToolType>();
  return dynamic_cast<ToolType*>(m_ToolManager->GetToolById(toolId));
}

}

#endif
