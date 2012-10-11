/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef _MIDASGENERALSEGMENTORVIEW_H_INCLUDED
#define _MIDASGENERALSEGMENTORVIEW_H_INCLUDED

#include "ui_MIDASGeneralSegmentorViewControls.h"

#include <QString>
#include <itkImage.h>
#include <itkImageRegion.h>
#include <itkPointSet.h>
#include <itkIndex.h>
#include <itkContinuousIndex.h>
#include <itkExtractImageFilter.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkOrthogonalContourExtractor2DImageFilter.h>
#include <itkPolyLineParametricPath.h>
#include <mitkPointSet.h>
#include <mitkSurface.h>
#include <mitkOperationActor.h>
#include <mitkOperation.h>
#include <mitkSliceNavigationController.h>

#include "QmitkMIDASBaseSegmentationFunctionality.h"
#include "itkMIDASHelper.h"
#include "itkMIDASRegionGrowingImageFilter.h"
#include "mitkMIDASContourTool.h"
#include "mitkMIDASDrawTool.h"
#include "mitkMIDASPolyTool.h"
#include "mitkMIDASToolKeyPressStateMachine.h"
#include "mitkMIDASToolKeyPressResponder.h"
#include "MIDASGeneralSegmentorViewControlsWidget.h"
#include "MIDASGeneralSegmentorViewCommands.h"
#include "MIDASGeneralSegmentorViewPipeline.h"
#include "MIDASGeneralSegmentorViewHelper.h"
#include "MIDASGeneralSegmentorViewPreferencePage.h"
#include "MIDASGeneralSegmentorViewEventInterface.h"

class QButtonGroup;
class QGridLayout;

/**
 * \class MIDASGeneralSegmentorView
 * \brief Provides the MIDAS hippocampus/ventricle segmentation developed at the Dementia Research Centre UCL.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 *
 * This class uses the mitk::ToolManager described in this paper on the
 * <a href="http://www.sciencedirect.com/science/article/pii/S0169260709001229">MITK Segmentation framework</a>.
 * The mitk::ToolManager has the following data sets registered in this order.
 * <pre>
 *   0. mitk::Image = the image being segmented, i.e. The Output.
 *   1. mitk::PointSet = the seeds for region growing.
 *   2. mitk::ContourSet = a set of contours for the current slice being edited - representing the current segmentation, i.e. green lines in MIDAS.
 *   3. mitk::ContourSet = a set of contours specifically for the draw tool, i.e. also green lines in MIDAS.
 *   4. mitk::ContourSet = a set of contours for the prior slice, i.e. whiteish lines in MIDAS.
 *   5. mitk::ContourSet = a set of contours for the next slice, i.e. turquoise blue lines in MIDAS.
 *   6. mitk::Image = binary image, same size as item 0, to represent the current region growing, i.e. blue lines in MIDAS.*
 * </pre>
 * and more specifically, items 1-6 are set up in the mitk::DataManager as hidden children of item 0.
 * Additional, significant bits of functionality include:
 *
 * <h2>Retain Marks</h2>
 *
 * The "retain marks" functionality only has an impact if we change slices. When the "retain marks"
 * checkbox is ticked, and we change slices we:
 * <pre>
 * 1. Check if the new slice is empty.
 * 2. If not empty we warn.
 * 3. If the user elects to overwrite the new slice, we simply copy all seeds and all image data to the new slice.
 * </pre>
 *
 * <h2>Recalculation of Seed Position</h2>
 *
 * The number of seeds for a slice often needs re-computing.  This is often because a slice
 * has been automatically propagated, and hence we need new seeds for each slice because
 * as you scroll through slices, regions without a seed would be wiped. For a given slice, the seeds
 * are set so that each disjoint (i.e. not-connected) region will have its own seed at the
 * largest minimum distance from the edge, scanning only in a vertical or horizontal direction.
 * In other words, for an image containing a single region:
 * <pre>
 * Find the first voxel in the image, best voxel location = current voxel location,
 * and best distance = maximum number of voxels along an image axis.
 * For each voxel
 *   Scan +x, -x, +y, -y and measure the minimum distance to the boundary
 *   If minimum distance > best distance
 *     best voxel location = current voxel location
 *     best distance = minimum distance
 * </pre>
 * The result is the largest minimum distance, or the largest distance to an edge, noting
 * that we are not scanning diagonally.
 *
 * <h2>Propagate Up/Down/3D</h2>
 *
 * Propagate runs a 3D region propagation from the current slice up/down, writing the
 * output to the current segmentation volume, overwriting anything already there.
 * The current slice is not affected. So, you can leave the threshold tick box either on or off.
 * For each subsequent slice in the up/down direction, the number of seeds is recomputed (as above).
 * 3D propagation is exactly equivalent to clicking "prop up" followed by "prop down".
 *
 * <h2>Threshold Apply</h2>
 *
 * The threshold "apply" button is only enabled when the threshold check-box is enabled,
 * and disabled otherwise. If MIDAS green contours, representing the current segmentation
 * are already present, then these contours limit the region growing.
 *
 * When we hit "apply":
 * <pre>
 * 1. Takes the current region growing image, and writes it to the current image.
 * 2. Recalculate the number of seeds for that slice, 1 per disjoint region, as above.
 * 3. Turn off thresholding, leaving sliders at current value.
 * </pre>
 * Undo therefore should:
 * <pre>
 * 1. Revert to the previous number of seeds, previous segmented region, previous contours,
 * 2. turn thresholding back on.
 * </pre>
 *
 * <h2>Wipe, Wipe+, Wipe-</h2>
 *
 * All three pieces of functionality appear similar, wiping the whole slice, whole anterior
 * region, or whole posterior region, including all segmentation and seeds. The threshold controls
 * are not changed. So, if it was on before, it will be on afterwards.
 *
 * \sa QmitkMIDASBaseSegmentationFunctionality
 * \sa MIDASMorphologicalSegmentorView
 */
class MIDASGeneralSegmentorView : public QmitkMIDASBaseSegmentationFunctionality,
                                  public mitk::MIDASToolKeyPressResponder,
                                  public mitk::OperationActor
{

  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  // A lot of the processing in this class is done with ITK,
  // and a lot of it is only relevant with unsigned char, and for 3D.
  typedef itk::Image<mitk::Tool::DefaultSegmentationDataType, 3> BinaryImage3DType;
  typedef BinaryImage3DType::RegionType                          Region3DType;
  typedef BinaryImage3DType::SizeType                            Size3DType;
  typedef BinaryImage3DType::IndexType                           Index3DType;
  typedef BinaryImage3DType::PointType                           Point3DType;
  typedef itk::Image<mitk::Tool::DefaultSegmentationDataType, 2> BinaryImage2DType;
  typedef BinaryImage2DType::RegionType                          Region2DType;
  typedef BinaryImage2DType::SizeType                            Size2DType;
  typedef BinaryImage2DType::IndexType                           Index2DType;
  typedef BinaryImage2DType::PointType                           Point2DType;
  typedef itk::ContinuousIndex<double, 2>                        ContinuousIndex2DType;
  typedef itk::ExtractImageFilter<BinaryImage3DType,
                                              BinaryImage2DType> ExtractSliceFilterType;
  typedef itk::OrthogonalContourExtractor2DImageFilter
                                 <BinaryImage2DType>             ExtractContoursFilterType;
  typedef itk::PolyLineParametricPath<2>                         PathType;

  /// \brief Constructor.
  MIDASGeneralSegmentorView();

  /// \brief Copy constructor which deliberately throws a runtime exception, as no-one should call it.
  MIDASGeneralSegmentorView(const MIDASGeneralSegmentorView& other);

  /// \brief Destructor.
  virtual ~MIDASGeneralSegmentorView();

  /// \brief Each View for a plugin has its own globally unique ID, this one is
  /// "uk.ac.ucl.cmic.midasgeneralsegmentor" and the .cpp file and plugin.xml should match.
  static const std::string VIEW_ID;

  /// \brief Returns the VIEW_ID = "uk.ac.ucl.cmic.midasgeneralsegmentor".
  virtual std::string GetViewID() const;

  /// \brief \see mitk::MIDASToolKeyPressResponder::SelectSeedTool()
  virtual bool SelectSeedTool();

  /// \brief \see mitk::MIDASToolKeyPressResponder::SelectDrawTool()
  virtual bool SelectDrawTool();

  /// \brief \see mitk::MIDASToolKeyPressResponder::UnselectTools()
  virtual bool UnselectTools();

  /// \brief \see mitk::MIDASToolKeyPressResponder::SelectPolyTool()
  virtual bool SelectPolyTool();

  /// \brief \see mitk::MIDASToolKeyPressResponder::SelectViewMode()
  virtual bool SelectViewMode();

  /// \brief \see mitk::MIDASToolKeyPressResponder::CleanSlice()
  virtual bool CleanSlice();

  /// \brief If the user hits the close icon, it is equivalent to a Cancel,
  /// and the segmentation is destroyed without warning.
  virtual void ClosePart();

  /// \brief Method to enable this class to interact with the Undo/Redo framework.
  virtual void ExecuteOperation(mitk::Operation* operation);

protected slots:
 
  /// \brief Qt slot called when the user hits the button "New segmentation",
  /// creating new working data such as a region growing image, a contour objects
  /// to store contour lines that we are drawing, and seeds for region growing.
  virtual mitk::DataNode* OnCreateNewSegmentationButtonPressed();

  /// \brief Qt slot called from the ToolManager when a segmentation tool is activated,
  /// toggles whether render windows listen to slice navigation events,
  /// i.e. it stops the cursors moving.
  virtual void OnToolSelected(int id);

  /// \brief Qt slot called from "see prior" checkbox to show the contour from the
  /// previous slice, this simply toggles the contours visibility.
  void OnSeePriorCheckBoxToggled(bool b);

  /// \brief Qt slot called from "see next" checkbox to show the contour from the next
  /// slice, this simply toggles the contours visibility.
  void OnSeeNextCheckBoxToggled(bool b);

  /// \brief Qt slot called from the "view" checkbox so that when b=true, we just
  /// see the image, when b=false, we additionally see all the contours, and again,
  /// this is just toggling data node visibility.
  void OnSeeImageCheckBoxPressed(bool b);

  /// \brief Qt slot called when the Clean button is pressed, indicating the
  /// current contours on the current slice should be cleaned, see additional spec,
  /// currently at:  https://cmicdev.cs.ucl.ac.uk/trac/ticket/1096
  void OnCleanButtonPressed();

  /// \brief Qt slot called when the Propagate Up button is pressed to take the
  /// current seeds and threshold values, and propagate Anterior/Superior/Right.
  ///
  /// This takes the seeds on the current slice, and for the region Anterior/Superior/Right
  /// will push the seeds into the first slice, and do 3D region growing, and calculate new
  /// seeds on a per slice basis.
  void OnPropagateUpButtonPressed();

  /// \brief Qt slot called when the Propagate Down button is pressed to take the current
  /// seeds and threshold values, and propagate Posterior/Inferor/Left.
  ///
  /// This takes the seeds on the current slice, and for the region Posterior/Inferor/Left
  /// will push the seeds into the first slice, and do 3D region growing, and calculate new
  /// seeds on a per slice basis.
  void OnPropagateDownButtonPressed();

  /// \brief Qt slot called when the Propagate 3D button is pressed that effectively calls
  /// OnPropagateUpButtonPressed() then OnPropagateDownButtonPressed() in that order.
  void OnPropagate3DButtonPressed();

  /// \brief Qt slot called when the Wipe button is pressed and will erase the current
  /// slice and seeds on the current slice.
  void OnWipeButtonPressed();

  /// \brief Qt slot called when the Wipe+ button is pressed and will erase the
  /// whole region Anterior/Superior/Right from the current slice, including seeds.
  void OnWipePlusButtonPressed();

  /// \brief Qt slot called when the Wipe- button is pressed and will erase the
  /// whole region Posterior/Inferior/Left from the current slice, including seeds.
  void OnWipeMinusButtonPressed();

  /// \brief Qt slot called when the OK button is pressed and accepts the current
  /// segmentation, destroying the working data (seeds, contours, region growing image),
  /// leaving you with a finished segmentation.
  void OnOKButtonPressed();

  /// \brief Qt slot called when the Reset button is pressed and resets to the start
  /// of the segmentation, so wipes the current segmentation (no undo), but leaves the
  /// reference data so you can continue segmenting.
  void OnResetButtonPressed();

  /// \brief Qt slot called when the Cancel button is pressed and destroys all working
  /// data (seeds, contours, region growing image), and also destroys the current segmentation.
  void OnCancelButtonPressed();

  /// \brief Qt slot called when the Apply button is pressed and used to accept the
  /// current segmentation, recalculates seed positions as per MIDAS spec described in
  /// this class intro.
  void OnThresholdApplyButtonPressed();

  /// \brief Qt slot called when the "threshold" checkbox is checked, and toggles
  /// the thresholding widget section on and calls MIDASGeneralSegmentorView::UpdateRegionGrowing.
  void OnThresholdCheckBoxToggled(bool b);

  /// \brief Qt slot called when the lower slider is moved, calls
  /// MIDASGeneralSegmentorView::UpdateRegionGrowing as thresholds have changed.
  void OnLowerThresholdValueChanged(double d);

  /// \brief Qt slot called when the upper slider is moved and calls
  /// MIDASGeneralSegmentorView::UpdateRegionGrowing as thresholds have changed.
  void OnUpperThresholdValueChanged(double d);

  /// \brief Qt slot called to effect a change of slice, which means accepting
  /// the current segmentation, and moving to the prior/next slice, see class intro.
  void OnSliceNumberChanged(int before, int after);

protected:

  /// \see mitk::ILifecycleAwarePart::PartVisible
  virtual void Visible();

  /// \see mitk::ILifecycleAwarePart::PartHidden
  virtual void Hidden();

  /// \brief Called by framework, this method creates all the controls for this view.
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Called by framework, this method can set the focus on a specific widget,
  /// but we currently do nothing.
  virtual void SetFocus();

  /// \brief Creates the connections of widgets in this class to the slots in this class.
  virtual void CreateConnections();

  /// \brief Method to enable derived classes to turn all widgets off/on to signify
  /// when the view is considered enabled/disabled.
  ///
  /// \see QmitkMIDASBaseSegmentation::EnableSegmentationWidgets
  virtual void EnableSegmentationWidgets(bool b);

  /// \brief For Irregular Volume Editing, a Segmentation image should have a grey
  /// scale parent, and several children as described in the class introduction.
  virtual bool IsNodeASegmentationImage(const mitk::DataNode::Pointer node);

  /// \brief We return true if the segmentation can be "re-started", i.e. you switch between binary images
  /// in the DataManager, and if the binary image has the correct hidden child nodes, then
  /// this returns true, indicating that it's a valid "in-progress" segmentation.
  virtual bool CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node);

  /// \brief Assumes input is a valid segmentation node, then searches for the derived
  /// children of the node, looking for the seeds and contours  as described in the class introduction.
  virtual mitk::ToolManager::DataVectorType GetWorkingNodesFromSegmentationNode(const mitk::DataNode::Pointer node);

  /// \brief Returns the name of the preferences node to look up.
  /// \see QmitkMIDASBaseSegmentationFunctionality::GetPreferencesNodeName
  virtual std::string GetPreferencesNodeName() { return MIDASGeneralSegmentorViewPreferencePage::PREFERENCES_NODE_NAME; }

  /// \brief This view registers with the mitk::DataStorage and listens for changing
  /// data, so this method is called when any node is changed, but only performs an update,
  /// if the nodes changed are those registered with the ToolManager as WorkingData,
  /// see class introduction.
  virtual void NodeChanged(const mitk::DataNode* node);

  /// \brief Called from the slice navigation controller to indicate a different slice,
  /// which in MIDAS terms means automatically accepting the currently segmented slice
  /// and moving to the next one.
  virtual void OnSliceChanged(const itk::EventObject & geometrySliceEvent);

  /// \brief Called from the registered Seed, Poly and Draw tools when the number of
  /// seeds has changed.
  virtual void OnNumberOfSeedsChanged(int numberOfSeeds);

  /// \brief Called from the registered Poly tool and Draw tool to indicate that
  /// contours have changed.
  virtual void OnContoursChanged();

private:

  // Operation constants, used in Undo/Redo framework
  static const mitk::OperationType OP_PROPAGATE;
  static const mitk::OperationType OP_THRESHOLD_APPLY;
  static const mitk::OperationType OP_WIPE;
  static const mitk::OperationType OP_CLEAN;
  static const mitk::OperationType OP_CHANGE_SLICE;
  static const mitk::OperationType OP_RETAIN_MARKS;

  /// \brief Utility method to check that we have initialised all the working data such as contours, region growing images etc.
  bool HaveInitialisedWorkingData();

  /// \brief Callback for when the window focus changes, where we update this view
  /// to be listening to the right window, and make sure ITK pipelines know we have
  /// changed orientation.
  void OnFocusChanged();

  /// \brief Used to create an image used for the region growing, see class intro.
  mitk::DataNode::Pointer CreateHelperImage(mitk::Image::Pointer referenceImage, mitk::DataNode::Pointer segmentationNode,  float r, float g, float b, std::string name, bool visible, int layer);

  /// \brief Used to create a contour set, used for the current, prior and next contours, see class intro.
  mitk::DataNode::Pointer CreateContourSet(mitk::DataNode::Pointer segmentationNode, float r, float g, float b, std::string name, bool visible, int layer);

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

  /// \brief Does propagate up/down, returning true if the propagation was performed and false
  /// otherwise.
  ///
  /// The return value is just used so that propagate3D can call the same method twice without
  /// displaying two warning messages.
  bool DoPropagate(bool showWarning, bool isUp, bool is3D);

  /// \brief Does wipe, where if direction=0, wipes current slice, if direction=1, wipes anterior,
  /// and if direction=-1, wipes posterior.
  bool DoWipe(int direction);

  /// \brief Method that actually does the threshold apply, so we can call it from the
  /// threshold apply button and not change slice, or when we change slice.
  bool DoThresholdApply(int oldSliceNumber, int newSliceNumber);

  /// \brief When the user draws on the slice, and puts seeds down, we may need to simply
  /// update the slice immediately.
  ///
  /// This function does not need to "Undo", as when you undo the adding of the seed or
  /// adding of any given contour, then you can simply recompute the current slice.
  void DoUpdateCurrentSlice();

  /// \brief Retrieves the lower and upper threshold from widgets and calls UpdateRegionGrowing.
  void UpdateRegionGrowing();

  /// \brief Given the two thresholds, and all seeds and contours, will recalculate the thresholded region in the current slice.
  /// \param isVisible whether the region growing volume should be visible.
  void UpdateRegionGrowing(bool isVisible, double lowerThreshold, double upperThreshold, bool skipUpdate);

  /// \brief Takes the current slice, and updates the prior (WorkingData[4]) and next (WorkingData[5]) contour sets.
  void UpdatePriorAndNext();

  /// \brief Takes the current slice, and refreshes the current slice contour set (WorkingData[2]).
  void UpdateCurrentSliceContours();

  /// \brief Takes the currently focussed window, and makes sure the segmented volume
  /// is not visible in the currently focussed window and takes the global visibility value in the previously
  /// focussed window, unless overrideToOn=true whereby both renderer specific properties are removed to revert to the global one.
  void UpdateSegmentationImageVisibility(bool overrideToGlobal);

  /// \brief Used to generate a contour outline round a binary segmentation image, and refreshes the outputSurface.
  ///
  /// Called for generating the "See Prior", "See Next" and also the outline contour of the current segmentation.
  void GenerateOutlineFromBinaryImage(mitk::Image::Pointer image,
      int axisNumber,
      int sliceNumber,
      int projectedSliceNumber,
      mitk::ContourSet::Pointer outputContourSet
      );

  /// \brief Clears both images of the working data.
  void ClearWorkingData();

  /// \brief Completely removes the current pipeline.
  void DestroyPipeline();

  /// \brief Removes the images we are using for editing during segmentation.
  void RemoveWorkingData();

  /// \brief Used to toggle tools on/off.
  void ToggleTool(int toolId);

  /// \brief Copies inputPoints to outputPoints
  void CopySeeds(const mitk::PointSet::Pointer inputPoints, mitk::PointSet::Pointer outputPoints);

  /// \brief Simply returns true if slice has any unenclosed seeds, and false otherwise.
  bool DoesSliceHaveUnenclosedSeeds();

  /**************************************************************
   * Start of ITK stuff.
   *************************************************************/

  /// \brief Fills the itkImage region with the fillValue.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKFillRegion(
      itk::Image<TPixel, VImageDimension>* itkImage,
      typename itk::Image<TPixel, VImageDimension>::RegionType &region,
      TPixel fillValue
      );


  /// \brief Clears an image by setting all voxels to zero.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKClearImage(
      itk::Image<TPixel, VImageDimension>* itkImage
      );


  /// \brief Copies an image, assumes input and output already allocated and of the same size.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKCopyImage(
      itk::Image<TPixel, VImageDimension>* input,
      itk::Image<TPixel, VImageDimension>* output
      );


  /// \brief Calculates the region corresponding to a single slice.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKCalculateSliceRegion(
      itk::Image<TPixel, VImageDimension>* itkImage,
      int axis,
      int slice,
      typename itk::Image<TPixel, VImageDimension>::RegionType &outputRegion
      );


  /// \brief Clears a slice by setting all voxels to zero for a given slice and axis.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKClearSlice(itk::Image<TPixel, VImageDimension>* itkImage,
      int axis,
      int slice
      );

  /// \brief Takes the inputSeeds and filters filters them so that outputSeeds
  /// contains just those seeds contained within the current slice.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKFilterSeedsToCurrentSlice(
      itk::Image<TPixel, VImageDimension> *itkImage,
      mitk::PointSet &inputSeeds,
      int axis,
      int slice,
      mitk::PointSet &outputSeeds
      );

  /// \brief Takes the inputSeeds and copies them to outputCopyOfInputSeeds,
  /// and also copies seeds to outputNewSeedsNotInRegionOfInterest if the seed
  /// is not within the region of interest.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKFilterInputPointSetToExcludeRegionOfInterest(
      itk::Image<TPixel, VImageDimension> *itkImage,
      typename itk::Image<TPixel, VImageDimension>::RegionType regionOfInterest,
      mitk::PointSet &inputSeeds,
      mitk::PointSet &outputCopyOfInputSeeds,
      mitk::PointSet &outputNewSeedsNotInRegionOfInterest
      );

  /// \brief Will return true if the given slice has seeds within that slice.
  template<typename TPixel, unsigned int VImageDimension>
  bool ITKSliceDoesHaveSeeds(
      itk::Image<TPixel, VImageDimension> *itkImage,
      mitk::PointSet* seeds,
      int axis,
      int slice
      );

  /// \brief Creates a region of interest within itkImage corresponding to the
  /// given slice, and checks if it is empty returning true if it is all zero.
  template<typename TPixel, unsigned int VImageDimension>
  bool ITKSliceIsEmpty(
      itk::Image<TPixel, VImageDimension> *itkImage,
      int axis,
      int slice,
      bool &outputSliceIsEmpty
      );

  /// \brief Called from RecalculateMinAndMaxOfSeedValues(), the actual method
  /// in ITK that recalculates the min and max intensity value of all the voxel
  /// locations given by the seeds.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKRecalculateMinAndMaxOfSeedValues(
      itk::Image<TPixel, VImageDimension>* itkImage,
      mitk::PointSet &inputSeeds,
      int axis,
      int slice,
      double &min,
      double &max
      );

  /// \brief Called from UpdateRegionGrowing(), updates the interactive ITK
  /// single 2D slice region growing pipeline.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKUpdateRegionGrowing(
      itk::Image<TPixel, VImageDimension> *itkImage,
      bool skipUpdate,
      mitk::Image &workingImage,
      mitk::PointSet &seeds,
      mitk::ContourSet &segmentationContours,
      mitk::ContourSet &drawContours,
      mitk::ContourSet &polyContours,
      itk::ORIENTATION_ENUM orientation,
      int sliceNumber,
      int axis,
      double lowerThreshold,
      double upperThreshold,
      mitk::DataNode::Pointer &outputRegionGrowingNode,
      mitk::Image::Pointer &outputRegionGrowingImage
      );

  /// \brief Method takes all the input, and calculates the 3D propagated
  /// region (up or down), and stores it in the region growing node.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKPropagateToRegionGrowingImage(
      itk::Image<TPixel, VImageDimension> *itkImage,
      mitk::PointSet &inputSeeds,
      itk::ORIENTATION_ENUM orientation,
      int sliceNumber,
      int axis,
      int direction,
      double lowerThreshold,
      double upperThreshold,
      mitk::PointSet &outputCopyOfInputSeeds,
      mitk::PointSet &outputNewSeeds,
      std::vector<int> &outputRegion,
      mitk::DataNode::Pointer &outputRegionGrowingNode,
      mitk::Image::Pointer &outputRegionGrowingImage
      );

  /// \brief Called from the ExecuteOperate (i.e. undo/redo framework) to
  /// actually apply the calculated propagated region to the current segmentation,
  /// and update the seeds accordingly.
  template <typename TGreyScalePixel, unsigned int VImageDimension>
  void ITKPropagateToSegmentationImage(
      itk::Image<TGreyScalePixel, VImageDimension>* referenceGreyScaleImage,
      mitk::Image* segmentedImage,
      mitk::Image* regionGrowingImage,
      mitk::PointSet* currentSeeds,
      mitk::OpPropagate *op);

  /// \brief Called to extract a contour set from a binary image, as might be used
  /// for "See Prior", "See Next", or the outlining a binary segmentation.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKGenerateOutlineFromBinaryImage(
      itk::Image<TPixel, VImageDimension>* itkImage,
      int axisNumber,
      int sliceNumber,
      int projectedSliceNumber,
      mitk::ContourSet::Pointer contourSet
      );

  /// \brief Works out the largest minimum distance to the edge of the image data, filtered on a given foregroundPixelValue.
  ///
  /// For each foreground voxel, search along the +/- x,y, (z if 3D) direction to find the minimum
  /// distance to the edge. Returns the largest minimum distance over the whole of the foreground region.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKGetLargestMinimumDistanceSeedLocation(
    itk::Image<TPixel, VImageDimension>* itkImage,
    TPixel& foregroundPixelValue,
    typename itk::Image<TPixel, VImageDimension>::IndexType &outputSeedIndex,
    int &outputDistance);

  /// \brief For the given input itkImage (assumed to always be binary), and regionOfInterest,
  /// will iterate on a slice by slice basis, recalculating new seeds.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKAddNewSeedsToPointSet(
      itk::Image<TPixel, VImageDimension> *itkImage,
      typename itk::Image<TPixel, VImageDimension>::RegionType regionOfInterest,
      int sliceNumber,
      int axisNumber,
      mitk::PointSet &outputNewSeeds
      );

  /// \brief Does any pre-processing of seeds necessary to facilitate Undo/Redo
  /// for Threshold Apply, and also changing slice.
  ///
  /// In this case means calculating the region of interest as a slice
  /// and if we are changing slice we propagate the seeds on the current slice to the new slice,
  /// and if we are doing threshold apply, we re-calculate seeds for the current slice based
  /// on the connected component analysis described in the class header at the top of this file.
  ///
  /// Notice how this is similar to the PreProcessing required for Propagate, seen in
  /// PropagateToRegionGrowingImageUsingITK. Also note that itkImage input should be the
  /// binary region growing node.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKPreProcessingOfSeedsForChangingSlice(
      itk::Image<TPixel, VImageDimension> *itkImage,
      mitk::PointSet &inputSeeds,
      int sliceNumber,
      int axis,
      int newSliceNumber,
      bool newSliceIsEmpty,
      mitk::PointSet &outputCopyOfInputSeeds,
      mitk::PointSet &outputNewSeeds,
      std::vector<int> &outputRegion
      );

  /// \brief Does any pre-processing necessary to facilitate Undo/Redo for Wipe commands,
  /// which in this case means computing a new list of seeds, and the region of interest to be wiped.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKPreProcessingForWipe(
      itk::Image<TPixel, VImageDimension> *itkImage,
      mitk::PointSet &inputSeeds,
      int sliceNumber,
      int axis,
      int direction,
      mitk::PointSet &outputCopyOfInputSeeds,
      mitk::PointSet &outputNewSeeds,
      std::vector<int> &outputRegion
      );

  /// \brief Does the wipe command for Wipe, Wipe+, Wipe-.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKDoWipe(
      itk::Image<TPixel, VImageDimension> *itkImage,
      mitk::PointSet* currentSeeds,
      mitk::OpWipe *op
      );

  /// \brief Returns true if the image has non-zero edge pixels, and false otherwise.
  template<typename TPixel, unsigned int VImageDimension>
  bool ITKImageHasNonZeroEdgePixels(
      itk::Image<TPixel, VImageDimension> *itkImage
      );

  /// \brief Will return true if slice has unenclosed seeds, and false otherwise.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKSliceDoesHaveUnEnclosedSeeds(
      itk::Image<TPixel, VImageDimension> *itkImage,
      mitk::PointSet &seeds,
      mitk::ContourSet &greenContours,
      mitk::ContourSet &yellowContours,
      double lowerThreshold,
      double upperThreshold,
      bool doRegionGrowing,
      int axis,
      int slice,
      bool &sliceDoesHaveUnenclosedSeeds
      );

  /// \brief Extracts a new contour set, for doing "Clean" operation.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKFilterContours(
      itk::Image<TPixel, VImageDimension> *itkImage,
      mitk::PointSet &seeds,
      mitk::ContourSet &greenContours,
      mitk::ContourSet &yellowContours,
      int axis,
      int slice,
      double lowerThreshold,
      double upperThreshold,
      mitk::ContourSet &outputContours
  );

  /// \brief Given an image, and a set of seeds, will append new seeds in the new slice if necessary.
  ///
  /// When MIDAS switches slice, if the current slice has seeds, and the new slice has none,
  /// it will auto-generate them. This is useful for things like quick region growing, as you
  /// simply switch slices, and the new region propagates forwards.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKPropagateSeedsToNewSlice(
      itk::Image<TPixel, VImageDimension> *itkImage,
      mitk::PointSet* currentSeeds,
      mitk::PointSet* newSeeds,
      int axis,
      int oldSliceNumber,
      int newSliceNumber
      );

  /// \brief Completely removes the current 2D region growing pipeline that is stored in the map m_TypeToPipelineMap.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKDestroyPipeline(
      itk::Image<TPixel, VImageDimension>* itkImage
      );


  /// \brief Called from InitialiseSeedsForVolume to create a seed for every distinct region on each slice.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKInitialiseSeedsForVolume(
      itk::Image<TPixel, VImageDimension> *itkImage,
      mitk::PointSet& seeds,
      int axis
      );

  /**************************************************************
   * End of ITK stuff.
   *************************************************************/

  /// \brief This class hooks into the Global Interaction system to respond to Key press events.
  mitk::MIDASToolKeyPressStateMachine::Pointer m_ToolKeyPressStateMachine;

  /// \brief Pointer to interface object, used as callback in Undo/Redo framework
  MIDASGeneralSegmentorViewEventInterface::Pointer m_Interface;

  /// \brief We hold a Map, containing a key comprised of the "typename TPixel, unsigned int VImageDimension"
  /// as a key, and the object containing the whole pipeline for single slice 2D region growing.
  typedef std::pair<std::string, GeneralSegmentorPipelineInterface*> StringAndPipelineInterfacePair;
  std::map<std::string, GeneralSegmentorPipelineInterface*> m_TypeToPipelineMap;

  /// \brief All the controls for the main view part.
  MIDASGeneralSegmentorViewControlsWidget* m_GeneralControls;

  /// \brief Used to put the base class widgets, and these widgets above in a common layout.
  QGridLayout *m_Layout;

  /// \brief Container for the Morphological Controls Widgets. \see QmitkMIDASBaseSegmentationFunctionality
  QWidget *m_ContainerForControlsWidget;

  /// \brief Keep track of this to SliceNavigationController register and unregister event listeners.
  mitk::SliceNavigationController::Pointer m_SliceNavigationController;

  /// \brief Each time the window changes, we register to the current slice navigation controller.
  unsigned long m_SliceNavigationControllerObserverTag;

  /// \brief Keep track of the previous slice number and reset to -1 when the window focus changes.
  int m_PreviousSliceNumber;

  /// \brief Used for the mitkFocusManager to register callbacks to track the currently focus window.
  unsigned long m_FocusManagerObserverTag;

  /// \brief Flag to stop re-entering code.
  bool m_IsUpdating;

};
#endif // _MIDASGENERALSEGMENTORVIEW_H_INCLUDED
