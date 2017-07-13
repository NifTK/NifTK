/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMorphologicalSegmentorController_h
#define niftkMorphologicalSegmentorController_h

#include <niftkMIDASGuiExports.h>

#include <niftkMorphologicalSegmentorPipelineManager.h>

#include <niftkBaseSegmentorController.h>


namespace niftk
{

class MorphologicalSegmentorGUI;

/// \class MorphologicalSegmentorController
class NIFTKMIDASGUI_EXPORT MorphologicalSegmentorController : public BaseSegmentorController
{
  Q_OBJECT

public:

  MorphologicalSegmentorController(IBaseView* view);
  virtual ~MorphologicalSegmentorController();

  /// \brief Sets up the GUI.
  /// This function has to be called from the CreateQtPartControl function of the view.
  virtual void SetupGUI(QWidget* parent) override;

  /// \brief If the user hits the close icon, it is equivalent to a Cancel,
  /// and the segmentation is destroyed without warning.
  void OnViewGetsClosed();

  /// \brief Called when the visibility of a data node has changed.
  virtual void OnNodeVisibilityChanged(const mitk::DataNode* node, const mitk::BaseRenderer* renderer) override;

  /// \brief Called when a data node has been removed.
  virtual void OnNodeRemoved(const mitk::DataNode* node) override;

  /// \brief Called when the segmentation is manually edited via the paintbrush tool.
  /// \param imageIndex tells which image has been modified: erosion addition / subtraction or dilation addition / subtraction.
  virtual void OnSegmentationEdited(int imageIndex);

protected:

  /// \brief For Morphological Editing, a Segmentation image should have a grey scale parent, and two binary children called SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME.
  virtual bool IsASegmentationImage(const mitk::DataNode* node) override;

  /// \brief For Morphological Editing, a Working image should be called either SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME, and have a binary image parent.
  virtual bool IsAWorkingImage(const mitk::DataNode* node) override;

  /// \brief Assumes input is a valid segmentation node, then searches for the derived children of the node, looking for binary images called SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME. Returns empty list if both not found.
  virtual std::vector<mitk::DataNode*> GetWorkingNodesFromSegmentationNode(mitk::DataNode* segmentationNode) override;

  /// \brief Creates the morphological segmentor widget that holds the GUI components of the view.
  virtual BaseGUI* CreateGUI(QWidget* parent) override;

  /// \brief Called when the reference data nodes have changed.
  virtual void OnReferenceNodesChanged() override;

  /// \brief Called when the working data nodes have changed.
  virtual void OnWorkingNodesChanged() override;

protected slots:

  /// \brief Called when the user hits the button "New segmentation", which creates the necessary reference data.
  virtual void OnNewSegmentationButtonClicked() override;

  /// \brief Sets the thresholding parameters.
  /// Called when the thresholding sliders or spin boxes changed.
  ///
  /// \param lowerThreshold the lowest intensity value included in the segmentation
  /// \param upperThreshold the upper intensity value included in the segmentation
  /// \param axialSliceNumber the number of the first slice, counting from the inferior end of the imaging volume to include in the imaging volume.
  void OnThresholdingValuesChanged(double lowerThreshold, double upperThreshold, int axialSliceNumber);

  /// \brief Sets the conditional erosion parameters.
  /// Called when the erosion sliders or spin boxes changed.
  ///
  /// \param upperThreshold the highest greyscale intensity value, above which the binary volume is not eroded
  /// \param numberOfErosions the number of erosion iterations to perform
  void OnErosionsValuesChanged(double upperThreshold, int numberOfErosions);

  /// \brief Sets the conditional dilation parameters.
  /// Called when the dilation sliders or spin boxes changed.
  ///
  /// \param lowerPercentage the lower percentage of the mean intensity value within the current region of interest, below which voxels are not dilated.
  /// \param upperPercentage the upper percentage of the mean intensity value within the current region of interest, below which voxels are not dilated.
  /// \param numberOfDilations the number of dilation iterations to perform
  void OnDilationsValuesChanged(double lowerPercentage, double upperPercentage, int numberOfDilations);

  /// \brief Sets the re-thresholding parameters.
  /// Called when re-thresholding widgets changed.
  ///
  /// \param boxSize the size of the re-thresholding box (see paper).
  void OnRethresholdingValuesChanged(int boxSize);

  /// \brief Called when we step to another stage of the pipeline, either fore or backwards.
  ///
  /// \param stage the new stage where we stepped to
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
  void SetControlsFromSegmentationNode();

  /// \brief All the GUI controls for the main Morphological Editor view part.
  MorphologicalSegmentorGUI* m_MorphologicalSegmentorGUI;

  /// \brief As much "business logic" as possible is delegated to this class so we can unit test it, without a GUI.
  MorphologicalSegmentorPipelineManager::Pointer m_PipelineManager;

  /// \brief Keep local variable to update after the tab has changed.
  int m_TabIndex;

};

}

#endif
