/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-05 16:22:23 +0000 (Mon, 05 Dec 2011) $
 Revision          : $Revision: 7920 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKMIDASMULTIVIEWEDITOR_H
#define QMITKMIDASMULTIVIEWEDITOR_H

#include <berryQtEditorPart.h>
#include <berryIPartListener.h>
#include <berryIPreferences.h>

#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>
#include <berryISelection.h>
#include <berryISelectionProvider.h>
#include <berryISelectionListener.h>

#include <QmitkDnDFrameWidget.h>
#include <uk_ac_ucl_cmic_gui_qt_common_Export.h>
#include "QmitkMIDASMultiViewWidget.h"
#include "QmitkMIDASMultiViewVisibilityManager.h"
#include "mitkDataStorage.h"
#include "mitkMIDASKeyPressStateMachine.h"

// CTK for event handling
#include "service/event/ctkEventHandler.h"
#include "service/event/ctkEventAdmin.h"

class ctkPluginContext;
struct ctkEventAdmin;

namespace mitk {
  class DataNode;
}

/**
 * \class QmitkMIDASMultiViewEditor
 * \brief Provides a MIDAS style layout, with up to 5 x 5 panes of equal size in a grid layout.
 * This class uses the ctkEventAdmin service to send and receive messages.
 *
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 */
class CMIC_QT_COMMON QmitkMIDASMultiViewEditor :
  public berry::QtEditorPart, public ctkEventHandler, virtual public berry::IPartListener
{
  Q_OBJECT
  Q_INTERFACES(ctkEventHandler)

public:

  berryObjectMacro(QmitkMIDASMultiViewEditor)
  QmitkMIDASMultiViewEditor();
  QmitkMIDASMultiViewEditor(const QmitkMIDASMultiViewEditor& other);
  ~QmitkMIDASMultiViewEditor();

  void Init(berry::IEditorSite::Pointer site, berry::IEditorInput::Pointer input);
  void DoSave() {}
  void DoSaveAs() {}
  bool IsDirty() const { return false; }
  bool IsSaveAsAllowed() const { return false; }

  static const std::string EDITOR_ID;

  /// \brief Tells the contained QmitkMIDASMultiViewWidget to setFocus().
  void SetFocus();

  /// \brief Get hold of the internal QmitkMIDASMultiViewWidget.
  QmitkMIDASMultiViewWidget* GetMIDASMultiViewWidget();

  /// \brief Called when the preferences object of this editor changed.
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

signals:

  /// \brief Signal that the value of the MIDAS controls (orientation, slice, magnification) should change.
  void UpdateMIDASViewingControlsValues(const ctkDictionary&);

  /// \brief Signal that the range of the values of the MIDAS controls (orientation, slice, magnification) should change.
  void UpdateMIDASViewingControlsRange(const ctkDictionary&);

public Q_SLOTS:

  /// \brief Handle events coming from the event admin service.
  void handleEvent(const ctkEvent& event);

  /// \brief This is received from QmitkMIDASMultiWidget when the window or image changes requiring to update the range of the controls on MIDASNavigationView.
  void OnUpdateMIDASViewingControlsRange(UpdateMIDASViewingControlsRangeInfo rangeInfo);

  /// \brief This is received from QmitkMIDASMultiWidget when the window or image changes requiring to update the value of the controls on MIDASNavigationView.
  void OnUpdateMIDASViewingControlsValues(UpdateMIDASViewingControlsInfo info);

protected:

  // Creates the main Qt GUI element parts.
  void CreateQtPartControl(QWidget* parent);

  // IPartListener
  Events::Types GetPartEventTypes() const;
  virtual void PartClosed (berry::IWorkbenchPartReference::Pointer partRef);
  virtual void PartHidden (berry::IWorkbenchPartReference::Pointer partRef);
  virtual void PartVisible (berry::IWorkbenchPartReference::Pointer partRef);

private:

  // Looks up the data storage service.
  mitk::DataStorage::Pointer GetDataStorage() const;

  // This class hooks into the Global Interaction system to respond to Key press events.
  mitk::MIDASKeyPressStateMachine::Pointer m_KeyPressStateMachine;

  // This class is the main central widget, containing multiple widgets such as rendering windows and control buttons.
  QmitkMIDASMultiViewWidget* m_MIDASMultiViewWidget;

  // This class is to manage visibility when nodes added, removed, main visibility properties changed etc. and manage the renderer specific properties.
  QmitkMIDASMultiViewVisibilityManager* m_MidasMultiViewVisibilityManager;

  // For Event Admin, we store a reference to the CTK plugin context
  ctkPluginContext* m_Context;

  // For Event Admin, we store a reference to the CTK event admin service
  ctkServiceReference m_EventAdminRef;

  // For Event Admin, we store a pointer to the actual CTK event admin implementation.
  ctkEventAdmin* m_EventAdmin;
};

#endif /*QMITKMIDASMULTIVIEWEDITOR_H*/
