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
#include <niftkImageOrientationUtils.h>

// Miscellaneous.
#include <mitkToolManager.h>
#include <itkImage.h>

class QmitkRenderWindow;
class niftkBaseSegmentorController;

/**
 * \class niftkBaseSegmentorView
 * \brief Base view component for MIDAS Segmentation widgets.
 *
 * \sa QmitkBaseView
 * \sa niftkMorphologicalSegmentorView
 * \sa niftkGeneralSegmentorView
 */
class CMIC_QT_COMMONMIDAS niftkBaseSegmentorView : public QmitkBaseView
{

  Q_OBJECT

public:

  niftkBaseSegmentorView();
  niftkBaseSegmentorView(const niftkBaseSegmentorView& other);
  virtual ~niftkBaseSegmentorView();

  /**
   * \brief Stores the preference name of the default outline colour (defaults to pure green).
   */
  static const QString DEFAULT_COLOUR;

  /**
   * \brief Stores the preference name of the default outline colour style sheet (defaults to pure green).
   */
  static const QString DEFAULT_COLOUR_STYLE_SHEET;

signals:

  /**
   * \brief Signal emmitted when we need to broadcast a request to turn interactors on/off.
   */
  void InteractorRequest(const ctkDictionary&);

protected:

  /**
   * \see mitk::ILifecycleAwarePart::PartActivated
   */
  virtual void Activated() override;

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
  niftk::ImageOrientation GetOrientationAsEnum();

  /// \brief Looks up the ReferenceImage registered with ToolManager and returns the axis [0,1,2] that corresponds to the given orientation, or -1 if it can't be found.
  int GetAxisFromReferenceImage(niftk::ImageOrientation orientation);

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

  /// \brief Creates the GUI parts.
  virtual void CreateQtPartControl(QWidget* parent) override;

  /// \brief Creates the segmentor controller that realises the GUI logic behind the view.
  virtual niftkBaseSegmentorController* CreateSegmentorController() = 0;

  /// \brief Decorates a DataNode according to the user preference settings, or requirements for binary images.
  void ApplyDisplayOptions(mitk::DataNode* node);

  /// \brief \see QmitkAbstractView::OnSelectionChanged.
  virtual void OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer>& nodes) override;

  /// \brief Called when preferences are updated.
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*) override;

  /// \brief Retrieve's the pref values from preference service, and store locally.
  virtual void RetrievePreferenceValues();

  /// \brief Derived classes decide which preferences are actually read.
  virtual QString GetPreferencesNodeName() = 0;

  /// \brief Returns the last selected node, whenever only a single node is selected. If you multi-select, this is not updated.
  mitk::DataNode::Pointer GetSelectedNode() const;

private:

  /// \brief The segmentor controller that realises the GUI logic behind the view.
  niftkBaseSegmentorController* m_SegmentorController;

friend class niftkBaseSegmentorController;

};

#endif
