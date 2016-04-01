/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkGeneralSegmentorView_h
#define niftkGeneralSegmentorView_h

#include <niftkBaseSegmentorView.h>

class niftkGeneralSegmentorController;
class niftkGeneralSegmentorGUI;

/**
 * \class niftkGeneralSegmentorView
 * \brief Provides a view for the MIDAS general purpose, Irregular Volume Editor functionality, originally developed
 * at the Dementia Research Centre UCL (http://dementia.ion.ucl.ac.uk/).
 *
 * \sa niftkBaseSegmentorView
 * \sa niftkGeneralSegmentorController
 * \sa niftkMorphologicalSegmentorView
 */
class niftkGeneralSegmentorView : public niftkBaseSegmentorView
{
  Q_OBJECT

public:

  /// \brief Constructor.
  niftkGeneralSegmentorView();

  /// \brief Copy constructor which deliberately throws a runtime exception, as no-one should call it.
  niftkGeneralSegmentorView(const niftkGeneralSegmentorView& other);

  /// \brief Destructor.
  virtual ~niftkGeneralSegmentorView();

  /// \brief Each View for a plugin has its own globally unique ID, this one is
  /// "uk.ac.ucl.cmic.midasgeneralsegmentor" and the .cxx file and plugin.xml should match.
  static const std::string VIEW_ID;

  /// \brief Returns the VIEW_ID = "uk.ac.ucl.cmic.midasgeneralsegmentor".
  virtual std::string GetViewID() const;

protected slots:
 
  /// \brief Qt slot called when the user hits the button "New segmentation",
  /// creating new working data such as a region growing image, contour objects
  /// to store contour lines that we are drawing, and seeds for region growing.
  virtual void OnNewSegmentationButtonClicked() override;

protected:

  /// \see mitk::ILifecycleAwarePart::PartVisible
  virtual void Visible() override;

  /// \see mitk::ILifecycleAwarePart::PartHidden
  virtual void Hidden() override;

  /// \brief Creates the general segmentor controller that realises the GUI logic behind the view.
  virtual niftkBaseSegmentorController* CreateSegmentorController() override;

  /// \brief Called by framework, this method can set the focus on a specific widget,
  /// but we currently do nothing.
  virtual void SetFocus() override;

  /// \brief Returns the name of the preferences node to look up.
  /// \see niftkBaseSegmentorView::GetPreferencesNodeName
  virtual QString GetPreferencesNodeName() override;

  /// \brief This view registers with the mitk::DataStorage and listens for changing
  /// data, so this method is called when any node is changed, but only performs an update,
  /// if the nodes changed are those registered with the ToolManager as WorkingData,
  /// see class introduction.
  virtual void NodeChanged(const mitk::DataNode* node) override;

  /// \brief This view registers with the mitk::DataStorage and listens for removing
  /// data, so this method cancels the operation and frees the resources if the
  /// segmentation node is removed.
  virtual void NodeRemoved(const mitk::DataNode* node) override;

  void onVisibilityChanged(const mitk::DataNode* node) override;

private:

  /// \brief Stores the initial state of the segmentation so that the Restart button can restore it.
  void StoreInitialSegmentation();

  /// \brief Used to create a contour set, used for the current, prior and next contours, see class intro.
  mitk::DataNode::Pointer CreateContourSet(mitk::DataNode::Pointer segmentationNode, float r, float g, float b, std::string name, bool visible, int layer);

  /// \brief Used to create an image used for the region growing, see class intro.
  mitk::DataNode::Pointer CreateHelperImage(mitk::Image::Pointer referenceImage, mitk::DataNode::Pointer segmentationNode,  float r, float g, float b, std::string name, bool visible, int layer);

  /// \brief Used when restarting a volume, to initialize all seeds for an existing segmentation.
  void InitialiseSeedsForWholeVolume();

  /// \brief Takes the current slice, and refreshes the current slice contour set (WorkingData[2]).
  void UpdateCurrentSliceContours(bool updateRendering=true);

  /// \brief Callback for when the window focus changes, where we update this view
  /// to be listening to the right window, and make sure ITK pipelines know we have
  /// changed orientation.
  void OnFocusChanged() override;

  /// \brief The general segmentor controller that realises the GUI logic behind the view.
  niftkGeneralSegmentorController* m_GeneralSegmentorController;

  /// \brief All the GUI controls for the main Irregular Editor view part.
  niftkGeneralSegmentorGUI* m_GeneralSegmentorGUI;

friend class niftkGeneralSegmentorController;

};

#endif
