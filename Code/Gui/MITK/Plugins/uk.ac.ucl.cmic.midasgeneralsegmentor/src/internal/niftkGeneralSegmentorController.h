/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkGeneralSegmentorController_h
#define __niftkGeneralSegmentorController_h

#include <mitkOperationActor.h>

#include <niftkBaseSegmentorController.h>
#include <niftkMIDASToolKeyPressResponder.h>

#include "niftkGeneralSegmentorEventInterface.h"

namespace mitk
{
class PointSet;
}

class niftkGeneralSegmentorGUI;
class niftkGeneralSegmentorView;

/**
 * \class niftkGeneralSegmentorController
 */
class niftkGeneralSegmentorController
  : public niftkBaseSegmentorController,
    public mitk::OperationActor,
    public niftk::MIDASToolKeyPressResponder
{

  Q_OBJECT

public:

  niftkGeneralSegmentorController(niftkGeneralSegmentorView* segmentorView);
  virtual ~niftkGeneralSegmentorController();

  /// \brief \see niftk::MIDASToolKeyPressResponder::SelectSeedTool()
  virtual bool SelectSeedTool() override;

  /// \brief \see niftk::MIDASToolKeyPressResponder::SelectDrawTool()
  virtual bool SelectDrawTool() override;

  /// \brief \see niftk::MIDASToolKeyPressResponder::UnselectTools()
  virtual bool UnselectTools() override;

  /// \brief \see niftk::MIDASToolKeyPressResponder::SelectPolyTool()
  virtual bool SelectPolyTool() override;

  /// \brief \see niftk::MIDASToolKeyPressResponder::SelectViewMode()
  virtual bool SelectViewMode() override;

  /// \brief \see niftk::MIDASToolKeyPressResponder::CleanSlice()
  virtual bool CleanSlice() override;

  /// \brief Method to enable this class to interact with the Undo/Redo framework.
  virtual void ExecuteOperation(mitk::Operation* operation) override;

protected:

  /// \brief For Irregular Volume Editing, a Segmentation image should have a grey
  /// scale parent, and several children as described in the class introduction.
  virtual bool IsNodeASegmentationImage(const mitk::DataNode::Pointer node) override;

  /// \brief Assumes input is a valid segmentation node, then searches for the derived
  /// children of the node, looking for the seeds and contours  as described in the class introduction.
  virtual mitk::ToolManager::DataVectorType GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer node) override;

  /// \brief We return true if the segmentation can be "re-started", i.e. you switch between binary images
  /// in the DataManager, and if the binary image has the correct hidden child nodes, then
  /// this returns true, indicating that it's a valid "in-progress" segmentation.
  virtual bool CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node) override;

    /// \brief Creates the general segmentor widget that holds the GUI components of the view.
  virtual niftkBaseSegmentorGUI* CreateSegmentorGUI(QWidget* parent) override;

protected slots:

  /// \brief Qt slot called from "see prior" checkbox to show the contour from the previous slice.
  void OnSeePriorCheckBoxToggled(bool checked);

  /// \brief Qt slot called from "see next" checkbox to show the contour from the next slice.
  void OnSeeNextCheckBoxToggled(bool checked);

  /// \brief Qt slot called when the Clean button is pressed, indicating the
  /// current contours on the current slice should be cleaned, see additional spec,
  /// currently at:  https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK/issues/1096
  void OnCleanButtonClicked();

private:

  /// \brief Used to create an image used for the region growing, see class intro.
  mitk::DataNode::Pointer CreateHelperImage(mitk::Image::Pointer referenceImage, mitk::DataNode::Pointer segmentationNode,  float r, float g, float b, std::string name, bool visible, int layer);

  /// \brief Used to create a contour set, used for the current, prior and next contours, see class intro.
  mitk::DataNode::Pointer CreateContourSet(mitk::DataNode::Pointer segmentationNode, float r, float g, float b, std::string name, bool visible, int layer);

  /// \brief Utility method to check that we have initialised all the working data such as contours, region growing images etc.
  bool HasInitialisedWorkingData();

  /// \brief Stores the initial state of the segmentation so that the Restart button can restore it.
  void StoreInitialSegmentation();

  /// \brief Looks for the Seeds registered as WorkingData[1] with the ToolManager.
  mitk::PointSet* GetSeeds();

  /// \brief Used when restarting a volume, to initialize all seeds for an existing segmentation.
  void InitialiseSeedsForWholeVolume();

  /// \brief Retrieves the min and max of the image (cached), and sets the thresholding
  /// intensity sliders range accordingly.
  void RecalculateMinAndMaxOfImage();

  /// \brief For each seed in the list of seeds and current slice, converts to millimetre position,
  /// and looks up the pixel value in the reference image (grey scale image being segmented)
  /// at that location, and updates the min and max labels on the GUI thresholding panel.
  void RecalculateMinAndMaxOfSeedValues();

  /// \brief Simply returns true if slice has any unenclosed seeds, and false otherwise.
  bool DoesSliceHaveUnenclosedSeeds(bool thresholdOn, int sliceNumber);

  /// \brief Simply returns true if slice has any unenclosed seeds, and false otherwise.
  bool DoesSliceHaveUnenclosedSeeds(bool thresholdOn, int sliceNumber, mitk::PointSet& seeds);

  /// \brief Filters seeds to current slice
  void FilterSeedsToCurrentSlice(
      mitk::PointSet& inputPoints,
      int& axis,
      int& sliceNumber,
      mitk::PointSet& outputPoints
      );

  /// \brief Filters seeds to current slice, and selects seeds that are enclosed.
  void FilterSeedsToEnclosedSeedsOnCurrentSlice(
      mitk::PointSet& inputPoints,
      bool& thresholdOn,
      int& sliceNumber,
      mitk::PointSet& outputPoints
      );

  /// \brief Retrieves the lower and upper threshold from widgets and calls UpdateRegionGrowing.
  void UpdateRegionGrowing(bool updateRendering = true);

  /// \brief Given the two thresholds, and all seeds and contours, will recalculate the thresholded region in the current slice.
  /// \param isVisible whether the region growing volume should be visible.
  void UpdateRegionGrowing(bool isVisible, int sliceNumber, double lowerThreshold, double upperThreshold, bool skipUpdate);

  /// \brief Takes the current slice, and refreshes the current slice contour set (WorkingData[2]).
  void UpdateCurrentSliceContours(bool updateRendering = true);

  /// \brief Takes the current slice, and updates the prior (WorkingData[4]) and next (WorkingData[5]) contour sets.
  void UpdatePriorAndNext(bool updateRendering = true);

  /// \brief Does wipe, where if direction=0, wipes current slice, if direction=1, wipes anterior,
  /// and if direction=-1, wipes posterior.
  bool DoWipe(int direction);

  /// \brief Method that actually does the threshold apply, so we can call it from the
  /// threshold apply button and not change slice, or when we change slice.
  bool DoThresholdApply(int oldSliceNumber, int newSliceNumber, bool optimiseSeeds, bool newSliceEmpty, bool newCheckboxStatus);

  /// \brief Used to toggle tools on/off.
  void ToggleTool(int toolId);

  /// \brief All the GUI controls for the main view part.
  niftkGeneralSegmentorGUI* m_GeneralSegmentorGUI;

  niftkGeneralSegmentorView* m_GeneralSegmentorView;

  /// \brief Pointer to interface object, used as callback in Undo/Redo framework
  niftkGeneralSegmentorEventInterface::Pointer m_Interface;

  /// \brief Flag to stop re-entering code, while updating.
  bool m_IsUpdating;

  /// \brief Flag to stop re-entering code, while trying to delete/clear the pipeline.
  bool m_IsDeleting;

  /// \brief Additional flag to stop re-entering code, specifically to block
  /// slice change commands from the slice navigation controller.
  bool m_IsChangingSlice;

  bool m_IsRestarting;

friend class niftkGeneralSegmentorView;

};

#endif
