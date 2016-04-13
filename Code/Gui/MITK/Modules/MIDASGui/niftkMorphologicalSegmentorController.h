/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkMorphologicalSegmentorController_h
#define __niftkMorphologicalSegmentorController_h

#include <niftkMIDASGuiExports.h>

#include <niftkMorphologicalSegmentorPipelineManager.h>

#include <niftkBaseSegmentorController.h>


class niftkMorphologicalSegmentorGUI;

/**
 * \class niftkMorphologicalSegmentorController
 */
class NIFTKMIDASGUI_EXPORT niftkMorphologicalSegmentorController : public niftkBaseSegmentorController
{
  Q_OBJECT

public:

  niftkMorphologicalSegmentorController(niftkIBaseView* view);
  virtual ~niftkMorphologicalSegmentorController();

  /// \brief Sets up the GUI.
  /// This function has to be called from the CreateQtPartControl function of the view.
  virtual void SetupSegmentorGUI(QWidget* parent) override;

  /// \brief If the user hits the close icon, it is equivalent to a Cancel,
  /// and the segmentation is destroyed without warning.
  void OnViewGetsClosed();

  /// \brief Called when a node is removed.
  virtual void OnNodeRemoved(const mitk::DataNode* node);

  /// \brief Called when the segmentation is manually edited via the paintbrush tool.
  /// \param imageIndex tells which image has been modified: erosion addition / subtraction or dilation addition / subtraction.
  virtual void OnSegmentationEdited(int imageIndex);

  void OnNodeVisibilityChanged(const mitk::DataNode* node);

protected:

  /// \brief For Morphological Editing, a Segmentation image should have a grey scale parent, and two binary children called SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME.
  virtual bool IsNodeASegmentationImage(const mitk::DataNode::Pointer node) override;

  /// \brief For Morphological Editing, a Working image should be called either SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME, and have a binary image parent.
  virtual bool IsNodeAWorkingImage(const mitk::DataNode::Pointer node) override;

  /// \brief Assumes input is a valid segmentation node, then searches for the derived children of the node, looking for binary images called SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME. Returns empty list if both not found.
  virtual mitk::ToolManager::DataVectorType GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer node) override;

  /// \brief Assumes input is a valid working node, then searches for a binary parent node, returns NULL if not found.
  virtual mitk::DataNode* GetSegmentationNodeFromWorkingData(const mitk::DataNode::Pointer node) override;

  /// \brief For any binary image, we return true if the property midas.morph.stage is present, and false otherwise.
  virtual bool CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node) override;

  /// \brief Creates the morphological segmentor widget that holds the GUI components of the view.
  virtual niftkBaseSegmentorGUI* CreateSegmentorGUI(QWidget* parent) override;

  /// \brief Called when the selection changes in the data manager.
  /// \see QmitkAbstractView::OnSelectionChanged.
  virtual void OnDataManagerSelectionChanged(const QList<mitk::DataNode::Pointer>& nodes) override;

protected slots:

  /// \brief Called when the user hits the button "New segmentation", which creates the necessary reference data.
  virtual void OnNewSegmentationButtonClicked() override;

  /// \brief Called from niftkMorphologicalSegmentorGUI when thresholding sliders or spin boxes changed.
  void OnThresholdingValuesChanged(double lowerThreshold, double upperThreshold, int axialSliceNumber);

  /// \brief Called from niftkMorphologicalSegmentorGUI when erosion sliders or spin boxes changed.
  void OnErosionsValuesChanged(double upperThreshold, int numberOfErosions);

  /// \brief Called from niftkMorphologicalSegmentorGUI when dilation sliders or spin boxes changed.
  void OnDilationsValuesChanged(double lowerPercentage, double upperPercentage, int numberOfDilations);

  /// \brief Called from niftkMorphologicalSegmentorGUI when re-thresholding widgets changed.
  void OnRethresholdingValuesChanged(int boxSize);

  /// \brief Called from niftkMorphologicalSegmentorGUI when a tab changes.
  void OnTabChanged(int i);

  /// \brief Called from niftkMorphologicalSegmentatorControls when OK button is clicked, which should finalise / finish and accept the segmentation.
  void OnOKButtonClicked();

  /// \brief Called from niftkMorphologicalSegmentatorControls when Restart button is clicked, which means "back to start", like a "reset" button.
  void OnRestartButtonClicked();

  /// \brief Called from niftkMorphologicalSegmentorGUI when cancel button is clicked, which should mean "throw away" / "abandon" current segmentation.
  void OnCancelButtonClicked();

private:

  /// \brief Creates a node for storing the axial cut-off plane.
  mitk::DataNode::Pointer CreateAxialCutOffPlaneNode(const mitk::Image* referenceImage);

  /// \brief Looks up the reference image, and sets default parameter values on the segmentation node.
  void SetSegmentationNodePropsFromReferenceImage();

  /// \brief Sets the morphological controls to default values specified by reference image, like min/max intensity range, number of axial slices etc.
  void SetControlsFromReferenceImage();

  /// \brief Sets the morphological controls by the current property values stored on the segmentation node.
  void SetControlsFromSegmentationNodeProps();

  /// \brief All the GUI controls for the main Morphological Editor view part.
  niftkMorphologicalSegmentorGUI* m_MorphologicalSegmentorGUI;

  /// \brief As much "business logic" as possible is delegated to this class so we can unit test it, without a GUI.
  niftk::MorphologicalSegmentorPipelineManager::Pointer m_PipelineManager;

  /// \brief Keep local variable to update after the tab has changed.
  int m_TabIndex;

friend class niftkMorphologicalSegmentorView;

};

#endif
