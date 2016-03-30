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

#include <niftkBaseSegmentorController.h>
#include <niftkMIDASToolKeyPressResponder.h>


namespace mitk
{
class PointSet;
}

class niftkGeneralSegmentorGUI;
class niftkGeneralSegmentorView;

/**
 * \class niftkGeneralSegmentorController
 */
class niftkGeneralSegmentorController : public niftkBaseSegmentorController, public niftk::MIDASToolKeyPressResponder
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

  /// \brief Takes the current slice, and refreshes the current slice contour set (WorkingData[2]).
  void UpdateCurrentSliceContours(bool updateRendering = true);

  /// \brief Takes the current slice, and updates the prior (WorkingData[4]) and next (WorkingData[5]) contour sets.
  void UpdatePriorAndNext(bool updateRendering = true);

  /// \brief Used to toggle tools on/off.
  void ToggleTool(int toolId);

  /// \brief All the GUI controls for the main view part.
  niftkGeneralSegmentorGUI* m_GeneralSegmentorGUI;

  niftkGeneralSegmentorView* m_GeneralSegmentorView;

friend class niftkGeneralSegmentorView;

};

#endif
