/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkBaseSegmentorView_h
#define __niftkBaseSegmentorView_h

#include <uk_ac_ucl_cmic_gui_qt_commonmidas_Export.h>

// CTK for event handling.
#include <service/event/ctkEventHandler.h>
#include <service/event/ctkEventAdmin.h>

// Berry stuff for application framework.
#include <berryIPreferences.h>
#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>

// Qmitk for Qt/MITK stuff.
#include <QmitkBaseView.h>
#include <niftkSegmentationSelectorWidget.h>
#include <niftkToolSelectorWidget.h>
#include <niftkMIDASOrientationUtils.h>

// Miscellaneous.
#include <mitkToolManager.h>
#include <itkImage.h>

#include <niftkMIDASEventFilter.h>

class QmitkRenderWindow;
class niftkBaseSegmentorController;
class niftkBaseSegmentorGUI;

/**
 * \class niftkBaseSegmentorView
 * \brief Base view component for MIDAS Segmentation widgets.
 *
 * \sa QmitkBaseView
 * \sa niftkMorphologicalSegmentorView
 * \sa niftkGeneralSegmentorView
 * \sa MITKSegmentationView
 */
class CMIC_QT_COMMONMIDAS niftkBaseSegmentorView : public QmitkBaseView, public niftk::MIDASEventFilter
{

  Q_OBJECT

public:

  niftkBaseSegmentorView();
  niftkBaseSegmentorView(const niftkBaseSegmentorView& other);
  virtual ~niftkBaseSegmentorView();

  /// \brief Returns true if the event should be filtered, i.e. not processed,
  /// otherwise false.
  virtual bool EventFilter(const mitk::StateEvent* stateEvent) const;

  /// \brief Returns true if the event should be filtered, i.e. not processed,
  /// otherwise false.
  virtual bool EventFilter(mitk::InteractionEvent* event) const;

  /**
   * \brief Stores the preference name of the default outline colour (defaults to pure green).
   */
  static const QString DEFAULT_COLOUR;

  /**
   * \brief Stores the preference name of the default outline colour style sheet (defaults to pure green).
   */
  static const QString DEFAULT_COLOUR_STYLE_SHEET;

  /**
   * \brief Creates from derived classes when the the user hits the "New segmentation", producing a dialog box,
   * and on successful completion of the dialog box, will create a new segmentation image.
   *
   * \param defaultColor The default colour to pass to the new segmentation dialog box.
   * \return mitk::DataNode* A new segmentation or <code>NULL</code> if the user cancells the dialog box.
   */
  virtual mitk::DataNode* CreateNewSegmentation(const QColor& defaultColor);

  /**
   * \brief Returns the currently focused renderer.
   *
   * Same as QmitkBaseView::GetFocusedRenderer(), but with public visiblity.
   *
   * \return mitk::BaseRenderer* The currently focused renderer, or NULL if it has not been set.
   */
  mitk::BaseRenderer* GetFocusedRenderer();

signals:

  /**
   * \brief Signal emmitted when we need to broadcast a request to turn interactors on/off.
   */
  void InteractorRequest(const ctkDictionary&);

protected slots:

  /**
   * \brief Called from niftkToolSelectorWidget when a tool changes.
   * We may need to enable or disable the editors from moving/changing position, zoom, etc.
   */
  virtual void OnToolSelected(int);

  /**
   * \brief Called from niftkSegmentationSelectorWidget when the 'Start/restart segmentation' button is clicked.
   */
  virtual void OnNewSegmentationButtonClicked();

protected:

  /**
   * \see mitk::ILifecycleAwarePart::PartActivated
   */
  virtual void Activated();

  /// \brief Gets a single binary image registered with the ToolManager (that tools can edit), or NULL if it can't be found or is not an image.
  mitk::Image* GetWorkingImageFromToolManager(int index);

  /// \brief Gets the reference node from the ToolManager or NULL if it can't be found.
  mitk::DataNode* GetReferenceNodeFromToolManager();

  /// \brief Gets the reference image from the ToolManager, or NULL if this doesn't yet exist or is not an image.
  mitk::Image* GetReferenceImageFromToolManager();

  /// \brief Assumes that the Reference (grey scale) node is always the direct parent of the Segmentation (binary) node, so we simply search for a non binary parent.
  mitk::DataNode* GetReferenceNodeFromSegmentationNode(const mitk::DataNode::Pointer node);

  /// \brief Assumes that a Reference (grey scale) image is ALWAYS registered with the ToolManager, so this method returns the reference image registered with the tool manager.
  mitk::Image* GetReferenceImage();

  /// \brief Works out the slice number.
  int GetSliceNumberFromSliceNavigationControllerAndReferenceImage();

  /// \brief Retrieves the currently active QmitkRenderWindow, and if it has a 2D mapper will return the current orientation of the view, returning ORIENTATION_UNKNOWN if it can't be found or the view is a 3D view for instance.
  MIDASOrientation GetOrientationAsEnum();

  /// \brief Looks up the ReferenceImage registered with ToolManager and returns the axis [0,1,2] that corresponds to the given orientation, or -1 if it can't be found.
  int GetAxisFromReferenceImage(const MIDASOrientation& orientation);

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

  /// \brief Makes sure the reference image is the selected one
  void SetReferenceImageSelected();

  /// \brief Returns the tool manager associated with this object (derived classes provide ones in different ways).
  mitk::ToolManager* GetToolManager();

  /// \brief Returns true if node represent an image that is non binary, and false otherwise.
  bool IsNodeAReferenceImage(const mitk::DataNode::Pointer node);

  /// \brief Returns true if node represents an image that is binary, and false otherwise.
  bool IsNodeASegmentationImage(const mitk::DataNode::Pointer node);

  /// \brief Subclasses decide if they can restart the segmentation for a binary node.
  bool CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node);

  /// \brief Returns true if node represents an image that is binary, and false otherwise.
  bool IsNodeAWorkingImage(const mitk::DataNode::Pointer node);

  /// \brief Assumes that a Working Node == a Segmentation Node, so simply returns the input node.
  mitk::ToolManager::DataVectorType GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer node);

  /// \brief Assumes that a Working Node == a Segmentation Node, so simply returns the input node.
  mitk::DataNode* GetSegmentationNodeFromWorkingData(const mitk::DataNode::Pointer node);

  /// \brief Gets a vector of the working data nodes (normally image, but could be surfaces etc) registered with the tool manager (ie. that tools can edit), or empty list if this can't be found.
  mitk::ToolManager::DataVectorType GetWorkingData();

  /// \brief Method to enable derived classes to turn widgets off/on, with default do nothing implementation.
  virtual void EnableSegmentationWidgets(bool enabled) = 0;

  /// \brief Turns the tool selection box on/off
  virtual void SetToolSelectorEnabled(bool enabled);

  /// \brief Creates the GUI parts.
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Creates the segmentor controller that realises the GUI logic behind the view.
  virtual niftkBaseSegmentorController* CreateSegmentorController() = 0;

  /// \brief Creates the segmentor widget that holds the GUI components of the view.
  /// This function is called from CreateQtPartControl. Derived classes should provide their implementation
  /// that returns an object whose class derives from niftkBaseSegmentorGUI.
  virtual niftkBaseSegmentorGUI* CreateSegmentorGUI(QWidget* parent) = 0;

  /// \brief Decorates a DataNode according to the user preference settings, or requirements for binary images.
  void ApplyDisplayOptions(mitk::DataNode* node);

  /// \brief Propagate BlueBerry selection to ToolManager for manual segmentation.
  virtual void SetToolManagerSelection(const mitk::DataNode* referenceData, const mitk::ToolManager::DataVectorType workingDataNodes);

  /// \brief \see QmitkAbstractView::OnSelectionChanged.
  virtual void OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes);

  /// \brief Called when preferences are updated.
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  /// \brief Retrieve's the pref values from preference service, and store locally.
  virtual void RetrievePreferenceValues();

  /// \brief Derived classes decide which preferences are actually read.
  virtual QString GetPreferencesNodeName() = 0;

  /// \brief Returns the last selected node, whenever only a single node is selected. If you multi-select, this is not updated.
  mitk::DataNode::Pointer GetSelectedNode() const;

  /// \brief Default colour to be displayed in the new segmentation dialog box.
  const QColor& GetDefaultSegmentationColor() const;

private:

  /// \brief The segmentor controller that realises the GUI logic behind the view.
  niftkBaseSegmentorController* m_SegmentorController;

  niftkBaseSegmentorGUI* m_SegmentorGUI;

  /// \brief Keeps track of the last selected node, whenever only a single node is selected. If you multi-select, this is not updated.
  mitk::DataNode::Pointer m_SelectedNode;

  /// \brief Keeps track of the last selected image, whenever only a single node is selected. If you multi-select, this is not updated.
  mitk::Image::Pointer m_SelectedImage;

  /// \brief Default colour to be displayed in the new segmentation dialog box.
  QColor m_DefaultSegmentationColor;

  /// \brief The ID of the currently active tool or -1 if no tool is active.
  int m_ActiveToolID;

  /// \brief Stores the visibility state of the cursor in the main display before activating a tool.
  bool m_MainWindowCursorVisibleWithToolsOff;

friend class niftkBaseSegmentorController;

};

#endif
