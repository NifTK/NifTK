/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkGeneralSegmentorController_h
#define niftkGeneralSegmentorController_h

#include <niftkMIDASGuiExports.h>

#include <mitkOperationActor.h>

#include <niftkBaseSegmentorController.h>
#include <niftkToolKeyPressResponder.h>
#include <niftkToolKeyPressStateMachine.h>

namespace mitk
{
class PointSet;
}

namespace niftk
{

class GeneralSegmentorControllerPrivate;
class GeneralSegmentorGUI;


/// \class GeneralSegmentorController
/// \brief Provides the MIDAS general purpose, Irregular Volume Editor functionality originally developed
/// at the Dementia Research Centre UCL (http://dementia.ion.ucl.ac.uk/).
///
/// This class uses the mitk::ToolManager and associated framework described in this paper on the
/// <a href="http://www.sciencedirect.com/science/article/pii/S0169260709001229">MITK Segmentation framework</a>.
///
/// The mitk::ToolManager has the following data sets registered in this order.
/// <pre>
///   0. mitk::Image = the image being segmented, i.e. The Output.
///   1. mitk::PointSet = the seeds for region growing, noting that the seeds are in 3D, spreading throughout the volume.
///   2. mitk::ContourModelSet = a set of contours for the current slice being edited - representing the current segmentation, i.e. green lines in MIDAS, but drawn here in orange.
///   3. mitk::ContourModelSet = a set of contours specifically for the draw tool, i.e. also green lines in MIDAS, and also drawn here in orange.
///   4. mitk::ContourModelSet = a set of contours for the prior slice, i.e. whiteish lines in MIDAS.
///   5. mitk::ContourModelSet = a set of contours for the next slice, i.e. turquoise blue lines in MIDAS.
///   6. mitk::Image = binary image, same size as item 0, to represent the current region growing, i.e. blue lines in MIDAS.
/// </pre>
/// Useful notes towards helping the understanding of this class
/// <ul>
///   <li>Items 1-6 are set up in the mitk::DataManager as hidden children of item 0.</li>
///   <li>The segmentation is very specific to a given view, as for example the ContourModelSet in WorkingData items 2,3,4,5 are only generated for a single slice, corresponding to the currently selected render window.</li>
///   <li>Region growing is 2D on the currently selected slice, except when doing propagate up or propagate down.</li>
///   <li>Apologies that this is rather a large monolithic class.</li>
/// </ul>
/// Additionally, significant bits of functionality include:
///
/// <h2>Recalculation of Seed Position</h2>
///
/// The number of seeds for a slice often needs re-computing.  This is often because a slice
/// has been automatically propagated, and hence we need new seeds for each slice because
/// as you scroll through slices, regions without a seed would be wiped. For a given slice, the seeds
/// are set so that each disjoint (i.e. not 4-connected) region will have its own seed at the
/// largest minimum distance from the edge, scanning only in a vertical or horizontal direction.
/// In other words, for an image containing a single region:
/// <pre>
/// Find the first voxel in the image, best voxel location = current voxel location,
/// and best distance = maximum number of voxels along an image slice axis.
/// For each voxel
///   Scan +x, -x, +y, -y and measure the minimum distance to the boundary
///   If minimum distance > best distance
///     best voxel location = current voxel location
///     best distance = minimum distance
/// </pre>
/// The result is the largest minimum distance, or the largest minimum distance to an edge, noting
/// that we are not scanning diagonally.
///
/// <h2>Propagate Up/Down/3D</h2>
///
/// Propagate runs a 3D region propagation from and including the current slice up/down, writing the
/// output to the current segmentation volume, overwriting anything already there.
/// The current slice is always affected. So, you can leave the threshold tick box either on or off.
/// For each subsequent slice in the up/down direction, the number of seeds is recomputed (as above).
/// 3D propagation is exactly equivalent to clicking "prop up" followed by "prop down".
/// Here, note that in 3D, you would normally do region growing in a 6-connected neighbourhood.
/// Here, we are doing a 5D connected neighbourhood, as you always propagate forwards in one
/// direction. i.e. in a coronal slice, and selecting "propagate up", which means propagate anterior,
/// then you cannot do region growing in the posterior direction. So its a 5D region growing.
///
/// <h2>Threshold Apply</h2>
///
/// The threshold "apply" button is only enabled when the threshold check-box is enabled,
/// and disabled otherwise. The current segmentation, draw tool contours and poly tool contours
/// (eg. WorkingData items 2 and 3, plus temporary data in the PolyTool) all limit the
/// region growing.
///
/// When we hit "apply":
/// <pre>
/// 1. Takes the current region growing image, and writes it to the current image.
/// 2. Recalculate the number of seeds for that slice, 1 per disjoint region, as above.
/// 3. Turn off thresholding, leaving sliders at current value.
/// </pre>
///
/// <h2>Wipe, Wipe+, Wipe-</h2>
///
/// All three pieces of functionality appear similar, wiping the whole slice, whole anterior
/// region, or whole posterior region, including all segmentation and seeds. The threshold controls
/// are not changed. So, if it was on before, it will be on afterwards.
///
/// <h2>Retain Marks</h2>
///
/// The "retain marks" functionality only has an impact if we change slices. When the "retain marks"
/// checkbox is ticked, and we change slices we:
/// <pre>
/// 1. Check if the new slice is empty.
/// 2. If not empty we warn.
/// 3. If the user elects to overwrite the new slice, we simply copy all seeds and all image data to the new slice.
/// </pre>
///
/// \sa niftkBaseSegmentorController
/// \sa MIDASMorphologicalSegmentorController
class NIFTKMIDASGUI_EXPORT GeneralSegmentorController
  : public BaseSegmentorController,
    public mitk::OperationActor,
    public ToolKeyPressResponder
{
  Q_OBJECT

public:

  GeneralSegmentorController(IBaseView* view);
  virtual ~GeneralSegmentorController();

  /// \brief Sets up the GUI.
  /// This function has to be called from the CreateQtPartControl function of the view.
  virtual void SetupGUI(QWidget* parent) override;

  virtual bool SelectSeedTool() override;

  virtual bool SelectDrawTool() override;

  virtual bool UnselectTools() override;

  virtual bool SelectPolyTool() override;

  virtual bool SelectViewMode() override;

  virtual bool CleanSlice() override;

  /// \brief Method to enable this class to interact with the Undo/Redo framework.
  virtual void ExecuteOperation(mitk::Operation* operation) override;

protected:

  /// \brief Assumes input is a valid segmentation node, then searches for the derived
  /// children of the node, looking for the seeds and contours  as described in the class introduction.
  virtual std::vector<mitk::DataNode*> GetWorkingNodesFrom(mitk::DataNode* segmentationNode) override;

  virtual bool IsNodeAValidReferenceImage(const mitk::DataNode* node) override;

  /// \brief Creates the general segmentor widget that holds the GUI components of the view.
  virtual BaseGUI* CreateGUI(QWidget* parent) override;

  virtual void OnNodeVisibilityChanged(const mitk::DataNode* node, const mitk::BaseRenderer* renderer) override;

  /// \brief Called when the reference data nodes have changed.
  virtual void OnReferenceNodesChanged() override;

  /// \brief Called when the working data nodes have changed.
  virtual void OnWorkingNodesChanged() override;

  /// \brief Called when the different slice gets selected in the viewer.
  /// This happens when a different renderer is selected or when the selected slice
  /// changes in the focused renderer either by interaction (e.g. scrolling by
  /// mouse wheel) or by API call.
  /// When the orientation changes, this function makes sure ITK pipelines know about that.
  /// Changing the selected slice in MIDAS terms means automatically accepting the currently
  /// segmented slice and moving to the next one, see class intro.
  /// \param orientation the orientation of the selected slice
  ///     It might not equal to the axis of the slice in the reference image.
  ///     This depends on the permutation of the axes.
  /// \param sliceIndex the index of the slice in the renderer (world space)
  ///     It might not equal to the index of the slice in the reference image.
  ///     This depends on the 'up direction' of the axis.
  virtual void OnSelectedSliceChanged(ImageOrientation orientation, int sliceIndex) override;

protected slots:

  /// \brief Qt slot called when the user hits the button "New segmentation",
  /// creating new working data such as a region growing image, contour objects
  /// to store contour lines that we are drawing, and seeds for region growing.
  virtual void OnNewSegmentationButtonClicked() override;

  /// \brief Qt slot called from "see prior" checkbox to show the contour from the previous slice.
  void OnSeePriorCheckBoxToggled(bool checked);

  /// \brief Qt slot called from "see next" checkbox to show the contour from the next slice.
  void OnSeeNextCheckBoxToggled(bool checked);

  /// \brief Qt slot called from "retain marks" checkbox.
  void OnRetainMarksCheckBoxToggled(bool checked);

  /// \brief Qt slot called when the Clean button is pressed, indicating the
  /// current contours on the current slice should be cleaned, see additional spec,
  /// currently at:  https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK/issues/1096
  void OnCleanButtonClicked();

  /// \brief Qt slot called when the Wipe button is pressed and will erase the current
  /// slice and seeds on the current slice.
  void OnWipeButtonClicked();

  /// \brief Qt slot called when the Wipe+ button is pressed and will erase the
  /// whole region Anterior/Superior/Right from the current slice, including seeds.
  void OnWipePlusButtonClicked();

  /// \brief Qt slot called when the Wipe- button is pressed and will erase the
  /// whole region Posterior/Inferior/Left from the current slice, including seeds.
  void OnWipeMinusButtonClicked();

  /// \brief Qt slot called when the Propagate Up button is pressed to take the
  /// current seeds and threshold values, and propagate Anterior/Superior/Right.
  void OnPropagateUpButtonClicked();

  /// \brief Qt slot called when the Propagate Down button is pressed to take the current
  /// seeds and threshold values, and propagate Posterior/Inferor/Left.
  void OnPropagateDownButtonClicked();

  /// \brief Qt slot called when the Propagate 3D button is pressed that is effectively
  /// equivalent to calling OnPropagateUpButtonPressed and OnPropagateDownButtonPressed.
  void OnPropagate3DButtonClicked();

  /// \brief Qt slot called when the Apply button is pressed and used to accept the
  /// current region growing segmentation, and recalculates seed positions as per MIDAS spec
  /// described in this class intro.
  void OnThresholdApplyButtonClicked();

  /// \brief Qt slot called when the "threshold" checkbox is checked, and toggles
  /// the thresholding widget section on and calls GeneralSegmentorController::UpdateRegionGrowing.
  void OnThresholdingCheckBoxToggled(bool checked);

  /// \brief Qt slot called when the lower or upper threshold slider is moved, calls
  /// GeneralSegmentorController::UpdateRegionGrowing as thresholds have changed.
  void OnThresholdValueChanged();

  /// \brief Qt slot called when the any button is pressed on this widget.
  ///
  /// It transfers the focus back to the main window so that the key interactions keep working.
  void OnAnyButtonClicked();

  /// \brief Qt slot called when the OK button is pressed and accepts the current
  /// segmentation, destroying the working data (seeds, contours, region growing image),
  /// leaving you with a finished segmentation.
  void OnOKButtonClicked();

  /// \brief Qt slot called when the Reset button is pressed and resets to the start
  /// of the segmentation, so wipes the current segmentation (no undo), but leaves the
  /// reference data so you can continue segmenting.
  void OnResetButtonClicked();

  /// \brief Qt slot called when the Cancel button is pressed and destroys all working
  /// data (seeds, contours, region growing image), and also destroys the current segmentation
  /// if it was created by this volume editor. Otherwise, it restores the original segmentation.
  void OnCancelButtonClicked();

  /// \brief Qt slot called when the Restart button is pressed and restores the initial
  /// state of the segmentation.
  void OnRestartButtonClicked();

private:

  /// \brief Returns which image coordinate corresponds to the currently selected orientation.
  /// Retrieves the currently active QmitkRenderWindow, and the reference image registered with the ToolManager,
  /// and returns the Image axis that the current view is looking along, or -1 if it can not be worked out.
  int GetReferenceImageSliceAxis();

  /// \brief Returns the slice index in the reference image that corresponds to the currently displayed slice.
  /// This might be different to the slice displayed in the viewer, depending on the up direction.
  int GetReferenceImageSliceIndex();

  /// \brief Returns the "Up" direction which is the anterior, superior or right direction depending on which orientation you are interested in.
  int GetReferenceImageSliceUpDirection();

  virtual void OnViewGetsVisible() override;

  virtual void OnViewGetsHidden() override;

  /// \brief If the user hits the close icon, it is equivalent to a Cancel,
  /// and the segmentation is destroyed without warning.
  void OnViewGetsClosed();

  /// \brief This view registers with the mitk::DataStorage and listens for changing
  /// data, so this method is called when any node is changed, but only performs an update,
  /// if the nodes changed are those registered with the ToolManager as WorkingData,
  /// see class introduction.
  virtual void OnNodeChanged(const mitk::DataNode* node) override;

  /// \brief This view registers with the mitk::DataStorage and listens for removing
  /// data, so this method cancels the operation and frees the resources if the
  /// segmentation node is removed.
  virtual void OnNodeRemoved(const mitk::DataNode* node) override;

  /// \brief Called from the registered Poly tool and Draw tool to indicate that contours have changed.
  virtual void OnContoursChanged();

  /// \brief Used to create an image used for the region growing, see class intro.
  mitk::DataNode::Pointer CreateHelperImage(const mitk::Image* referenceImage, const mitk::Color& colour, const std::string& name, bool visible, int layer);

  /// \brief Used to create a contour set, used for the current, prior and next contours, see class intro.
  mitk::DataNode::Pointer CreateContourSet(const mitk::Color& colour, const std::string& name, bool visible, int layer);

  /// \brief Stores the initial state of the segmentation so that the Restart button can restore it.
  void StoreInitialSegmentation();

  /// \brief Looks for the Seeds registered as WorkingData[1] with the ToolManager.
  mitk::PointSet* GetSeeds();

  /// \brief Initialises seeds for a given slice.
  /// Used when starting a segmentation or switching orientation, to place seeds
  /// into the regions of the current slice.
  void InitialiseSeedsForSlice(int sliceAxis, int sliceIndex);

  /// \brief For each seed in the list of seeds and current slice, converts to millimetre position,
  /// and looks up the pixel value in the reference image (grey scale image being segmented)
  /// at that location, and updates the min and max labels on the GUI thresholding panel.
  void RecalculateMinAndMaxOfSeedValues();

  /// \brief Simply returns true if slice has any unenclosed seeds, and false otherwise.
  bool DoesSliceHaveUnenclosedSeeds(bool thresholdOn, int sliceAxis, int sliceIndex);

  /// \brief Simply returns true if slice has any unenclosed seeds, and false otherwise.
  bool DoesSliceHaveUnenclosedSeeds(bool thresholdOn, int sliceAxis, int sliceIndex, const mitk::PointSet* seeds);

  /// \brief Filters seeds to current slice
  void FilterSeedsToCurrentSlice(
      const mitk::PointSet* inputPoints,
      int sliceAxis,
      int sliceIndex,
      mitk::PointSet* outputPoints
      );

  /// \brief Filters seeds to slice, and selects seeds that are enclosed.
  void FilterSeedsToEnclosedSeedsOnSlice(
      const mitk::PointSet* inputPoints,
      bool thresholdOn,
      int sliceAxis,
      int sliceIndex,
      mitk::PointSet* outputPoints
      );

  /// \brief Retrieves the lower and upper threshold from widgets and calls UpdateRegionGrowing.
  void UpdateRegionGrowing(bool updateRendering = true);

  /// \brief Given the two thresholds, and all seeds and contours, will recalculate the thresholded region in the given slice.
  /// \param isVisible whether the region growing volume should be visible.
  void UpdateRegionGrowing(bool isVisible, int sliceAxis, int sliceIndex, double lowerThreshold, double upperThreshold, bool skipUpdate);

  /// \brief Takes the current slice, and refreshes the current slice contour set (WorkingData[2]).
  void UpdateCurrentSliceContours(bool updateRendering = true);

  /// \brief Takes the current slice, and updates the prior (WorkingData[4]) and next (WorkingData[5]) contour sets.
  void UpdatePriorAndNext(bool updateRendering = true);

  /// \brief Does propagate up/down/3D.
  void DoPropagate(bool isUp, bool is3D);

  /// \brief Does wipe, where if direction=0, wipes current slice, if direction=1, wipes anterior,
  /// and if direction=-1, wipes posterior.
  void DoWipe(int direction);

  /// \brief Method that actually does the threshold apply, so we can call it from the
  /// threshold apply button and not change slice, or when we change slice.
  /// It applies the threshold on the current slice.
  void DoThresholdApply(bool optimiseSeeds, bool newSliceEmpty, bool newCheckboxStatus);

  /// \brief Used to toggle tools on/off.
  void ToggleTool(int toolId);

  /// \brief Completely removes the current pipeline.
  void DestroyPipeline();

  /// \brief Removes the images we are using for editing during segmentation.
  void RemoveWorkingNodes();

  /// \brief Restores the initial state of the segmentation after the Restart button was pressed.
  void RestoreInitialSegmentation();

  /// \brief Called when the view is closed or the segmentation node is removed from the data
  /// manager and destroys all working data (seeds, contours, region growing image), and also
  /// destroys the current segmentation.
  void DiscardSegmentation();

  /// \brief Clears both images of the working data.
  void ClearWorkingNodes();

  QScopedPointer<GeneralSegmentorControllerPrivate> d_ptr;

  Q_DECLARE_PRIVATE(GeneralSegmentorController)

};

}

#endif
