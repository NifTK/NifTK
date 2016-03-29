/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkMorphologicalSegmentorView_h
#define __niftkMorphologicalSegmentorView_h

#include <niftkBaseSegmentorView.h>

#include <mitkImage.h>

#include <MorphologicalSegmentorPipelineParams.h>
#include "niftkMorphologicalSegmentorPreferencePage.h"
#include <niftkMorphologicalSegmentorPipelineManager.h>

class niftkMorphologicalSegmentorController;
class niftkMorphologicalSegmentorGUI;

/**
 * \class niftkMorphologicalSegmentorView
 * \brief Provides the plugin component for the MIDAS brain segmentation functionality, originally developed at the Dementia Research Centre UCL.
 *
 * This plugin implements the paper:
 *
 * "Interactive algorithms for the segmentation and quantification of 3-D MRI brain scans"
 * by P. A. Freeborough, N. C. Fox and R. I. Kitney, published in
 * Computer Methods and Programs in Biomedicine 53 (1997) 15-25.
 *
 * \ingroup uk_ac_ucl_cmic_midasmorphologicalsegmentor_internal
 *
 * \sa niftkBaseSegmentorView
 * \sa niftkMorphologicalSegmentorPipelineManager
 * \sa MorphologicalSegmentorPipeline
 * \sa MorphologicalSegmentorPipelineInterface
 * \sa MorphologicalSegmentorPipelineParams
 */
class niftkMorphologicalSegmentorView : public niftkBaseSegmentorView
{

  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  /// \brief Constructor, but most GUI construction is done in CreateQtPartControl().
  niftkMorphologicalSegmentorView();

  /// \brief Copy constructor which deliberately throws a runtime exception, as no-one should call it.
  niftkMorphologicalSegmentorView(const niftkMorphologicalSegmentorView& other);

  /// \brief Destructor.
  virtual ~niftkMorphologicalSegmentorView();

  /// \brief Each View for a plugin has its own globally unique ID.
  static const std::string VIEW_ID;

  /// \brief Returns VIEW_ID = uk.ac.ucl.cmic.midasmorphologicalsegmentor.
  virtual std::string GetViewID() const;

  /// \brief If the user hits the close icon, it is equivalent to a Cancel.
  virtual void ClosePart();

protected slots:
 
  /// \brief Called when the user hits the button "New segmentation", which creates the necessary reference data.
  virtual void OnNewSegmentationButtonClicked();

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

protected:

  /// \brief Called by framework, this method creates all the controls for this view
  virtual void CreateQtPartControl(QWidget *parent) override;

  /// \brief Creates the morphological segmentor controller that realises the GUI logic behind the view.
  virtual niftkBaseSegmentorController* CreateSegmentorController() override;

  /// \brief Creates the morphological segmentor widget that holds the GUI components of the view.
  virtual niftkBaseSegmentorGUI* CreateSegmentorGUI(QWidget* parent) override;

  /// \brief Called by framework, sets the focus on a specific widget, but currently does nothing.
  virtual void SetFocus() override;

  /// \brief Method to enable this and derived classes to turn widgets off/on
  virtual void EnableSegmentationWidgets(bool enabled) override;

  /// \brief Called when a node is removed.
  virtual void NodeRemoved(const mitk::DataNode* node) override;

  /// \brief Returns the name of the preferences node to look up.
  virtual QString GetPreferencesNodeName() override;

  /// \brief \see QmitkAbstractView::OnSelectionChanged.
  virtual void OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes) override;

  void onVisibilityChanged(const mitk::DataNode* node) override;

private:

  /// \brief Creates a node for storing the axial cut-off plane.
  mitk::DataNode::Pointer CreateAxialCutOffPlaneNode(const mitk::Image* referenceImage);

  /// \brief Looks up the reference image, and sets default parameter values on the segmentation node.
  void SetSegmentationNodePropsFromReferenceImage();

  /// \brief Sets the morphological controls to default values specified by reference image, like min/max intensity range, number of axial slices etc.
  void SetControlsFromReferenceImage();

  /// \brief Sets the morphological controls by the current property values stored on the segmentation node.
  void SetControlsFromSegmentationNodeProps();

  /// \brief Called when the segmentation is manually edited via the paintbrush tool.
  /// \param imageIndex tells which image has been modified: erosion addition / subtraction or dilation addition / subtraction.
  virtual void OnSegmentationEdited(int imageIndex);

  /// \brief The morphological segmentor controller that realises the GUI logic behind the view.
  niftkMorphologicalSegmentorController* m_MorphologicalSegmentorController;

  /// \brief All the GUI controls for the main Morphological Editor view part.
  niftkMorphologicalSegmentorGUI* m_MorphologicalSegmentorGUI;

  /// \brief As much "business logic" as possible is delegated to this class so we can unit test it, without a GUI.
  niftk::MorphologicalSegmentorPipelineManager::Pointer m_PipelineManager;

  /// \brief Keep local variable to update after the tab has changed.
  int m_TabIndex;

friend class niftkMorphologicalSegmentorController;

};

#endif
