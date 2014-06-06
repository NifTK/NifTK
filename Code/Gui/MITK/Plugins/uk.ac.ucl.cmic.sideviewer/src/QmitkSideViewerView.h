/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkSideViewerView_h
#define QmitkSideViewerView_h

#include <uk_ac_ucl_cmic_sideviewer_Export.h>

// CTK for event handling.
#include <service/event/ctkEventHandler.h>
#include <service/event/ctkEventAdmin.h>

// Berry stuff for application framework.
#include <berryIPreferences.h>
#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>

// Qmitk for Qt/MITK stuff.
#include <QmitkBaseView.h>
#include "QmitkSideViewerWidget.h"

#include <mitkMIDASEventFilter.h>

class QmitkRenderWindow;

/**
 * \class QmitkSideViewerView
 * \brief Base view component for MIDAS Segmentation widgets.
 *
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 *
 * \sa QmitkBaseView
 */
class CMIC_QT_SIDEVIEWER QmitkSideViewerView : public QmitkBaseView, public mitk::MIDASEventFilter
{

  Q_OBJECT

public:

  QmitkSideViewerView();
  QmitkSideViewerView(const QmitkSideViewerView& other);
  virtual ~QmitkSideViewerView();

  /// \brief Returns true if the event should be filtered, i.e. not processed,
  /// otherwise false.
  virtual bool EventFilter(const mitk::StateEvent* stateEvent) const;

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

protected:

  /**
   * \see mitk::ILifecycleAwarePart::PartActivated
   */
  virtual void Activated();

  /**
   * \see mitk::ILifecycleAwarePart::PartDectivated
   */
  virtual void Deactivated();

  /**
   * \see mitk::ILifecycleAwarePart::PartVisible
   */
  virtual void Visible();

  /**
   * \see mitk::ILifecycleAwarePart::PartHidden
   */
  virtual void Hidden();

  /// \brief Creates the GUI parts.
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Called by framework, sets the focus on a specific widget.
  virtual void SetFocus();

  /// \brief Decorates a DataNode according to the user preference settings, or requirements for binary images.
  virtual void ApplyDisplayOptions(mitk::DataNode* node);

  /// \brief Called when preferences are updated.
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  /// \brief Retrieve's the pref values from preference service, and store locally.
  virtual void RetrievePreferenceValues();

  /// \brief Derived classes decide which preferences are actually read.
  virtual std::string GetPreferencesNodeName();

  /// \brief Provides an additional view of the segmented image, so plugin can be used on second monitor.
  QmitkSideViewerWidget *m_SideViewerWidget;

private:

  /// \brief For Event Admin, we store a reference to the CTK plugin context
  ctkPluginContext* m_Context;

  /// \brief For Event Admin, we store a reference to the CTK event admin service
  ctkServiceReference m_EventAdminRef;

  /// \brief For Event Admin, we store a pointer to the actual CTK event admin implementation.
  ctkEventAdmin* m_EventAdmin;

};
#endif
