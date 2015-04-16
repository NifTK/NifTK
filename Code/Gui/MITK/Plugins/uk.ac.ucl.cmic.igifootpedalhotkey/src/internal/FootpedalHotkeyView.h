/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef FootpedalHotkeyView_h
#define FootpedalHotkeyView_h

#include <QmitkBaseView.h>
#include <service/event/ctkEvent.h>
#include "ui_FootpedalHotkeyViewWidget.h"
#include <QTimer>


class QmitkWindowsHotkeyHandler;

/**
 * Reacts to presses on a foot pedal.
 * The pedals generate keystrokes that are registered as system-wide hotkeys, i.e. no matter
 * what window currently has focus, the foot pedal presses will always be routed to this plugin.
 * It then sends off a CTK event to trigger an action in some other plugin.
 */
class FootpedalHotkeyView : public QmitkBaseView, public Ui::FootpedalHotkeyViewWidget
{  
  Q_OBJECT

public:

  FootpedalHotkeyView();
  virtual ~FootpedalHotkeyView();

  /**
   * \brief Static view ID = uk.ac.ucl.cmic.igifootpedalhotkey
   */
  static const char* VIEW_ID;

  /**
   * \brief Returns the view ID.
   */

  virtual std::string GetViewID() const;

signals:
  void OnStartRecording(ctkDictionary d);
  void OnStopRecording(ctkDictionary d);

protected:

  /**
   *  \brief Called by framework, this method creates all the controls for this view
   */
  virtual void CreateQtPartControl(QWidget *parent);

  /**
   * \brief Called by framework, sets the focus on a specific widget.
   */
  virtual void SetFocus();


  /** Called via ctk-event bus when user starts an IGI data recording session. */
  void WriteCurrentConfig(const QString& directory) const;

protected slots:


private slots:

  /** Triggered by igidatasources plugin (and QmitkIGIDataSourceManager) to tell us that recording has started. */
  void OnRecordingStarted(const ctkEvent& event);

  void OnHotkeyPressed(QmitkWindowsHotkeyHandler* sender, int hotkey);

  void OnTimer1();
  void OnTimer2();
  void OnTimer3();

private:
  QmitkWindowsHotkeyHandler*      m_Footswitch1;
  QTimer*                         m_Footswitch1OffTimer;    // to emulate hotkey-release events.
  QmitkWindowsHotkeyHandler*      m_Footswitch2;
  QTimer*                         m_Footswitch2OffTimer;
  QmitkWindowsHotkeyHandler*      m_Footswitch3;
  QTimer*                         m_Footswitch3OffTimer;


  // these are coming from the ctk event bus admin. we use them to explicitly unregister ourself.
  qlonglong           m_IGIRecordingStartedSubscriptionID;
};

#endif // FootpedalHotkeyView_h
