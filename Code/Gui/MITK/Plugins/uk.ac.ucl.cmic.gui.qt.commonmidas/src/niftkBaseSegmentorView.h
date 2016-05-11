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
#include <berryIWorkbenchPart.h>

// Qmitk for Qt/MITK stuff.
#include <QmitkBaseView.h>

namespace berry
{
class IBerryPreferences;
}

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

  /// \brief Creates the GUI parts.
  virtual void CreateQtPartControl(QWidget* parent) override;

  /// \brief Creates the segmentor controller that realises the GUI logic behind the view.
  virtual niftkBaseSegmentorController* CreateSegmentorController() = 0;

  /// \brief \see QmitkAbstractView::OnSelectionChanged.
  virtual void OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer>& nodes) override;

  /// \brief Called when preferences are updated.
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*) override;

  /// \brief Retrieve's the pref values from preference service, and store locally.
  virtual void RetrievePreferenceValues();

  /// \brief Derived classes decide which preferences are actually read.
  virtual QString GetPreferencesNodeName() = 0;

  void onVisibilityChanged(const mitk::DataNode* node) override;

  /// \brief This view registers with the mitk::DataStorage and listens for changing
  /// data, so this method is called when any node is changed, but only performs an update,
  /// if the nodes changed are those registered with the ToolManager as WorkingData,
  /// see class introduction.
  virtual void NodeChanged(const mitk::DataNode* node) override;

  /// \brief This view registers with the mitk::DataStorage and listens for removing
  /// data, so this method cancels the operation and frees the resources if the
  /// segmentation node is removed.
  virtual void NodeRemoved(const mitk::DataNode* node) override;

private:

  /// \brief The segmentor controller that realises the GUI logic behind the view.
  niftkBaseSegmentorController* m_SegmentorController;

};

#endif
