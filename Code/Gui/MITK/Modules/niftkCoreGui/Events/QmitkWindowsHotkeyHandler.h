/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkWindowsHotkeyHandler_h
#define QmitkWindowsHotkeyHandler_h

#include <niftkCoreGuiExports.h>
#include <QThread>


// pimpl to hide all the nasty windows internals.
struct QmitkWindowsHotkeyHandlerImpl;


/**
 * Custom thread with low-level Windows event loop to handle hotkey messages.
 * These are not processed by Qt and simply disappear in its event handling mechanism.
 * Works only on Windows.
 * Instances of this object should be created by the main event loop. Do not
 * moveToThread() this object to itself!
 */
class NIFTKCOREGUI_EXPORT QmitkWindowsHotkeyHandler : public QThread
{
  Q_OBJECT

public:
  /**
   * Currently known hotkeys.
   * Note that the value is important: it decodes to virtual key codes etc.
   */
  enum Hotkey
  {
    CTRL_ALT_F5     = 0x00030074    // VK_F5 | ((MOD_ALT | MOD_CONTROL) << 16)
  };


  /**
   * Creates a new handler and starts the corresponding thread.
   * Note that you have to know in advance which hotkey you want, there is no
   * way to change it afterwards.
   *
   * Init happens in run(), not in this constructor. Unfortunately, this means that
   * error reporting is practically non-existent.
   *
   * @throws nothing should not throw anything.
   */
  QmitkWindowsHotkeyHandler(Hotkey hk);
  virtual ~QmitkWindowsHotkeyHandler();


signals:
  /**
   * Beware: you really should connect this as QueuedConnection!
   */
  void HotkeyPressed(QmitkWindowsHotkeyHandler* sender, int hotkey);


protected:
  /**
   * Does the initialisation and runs a low-level Windows message loop.
   * This will not do any Qt signal delivery! So it's important that you do
   * not moveToThread() this object.
   */
  virtual void run();


private:
  QmitkWindowsHotkeyHandler(const QmitkWindowsHotkeyHandler& copyme);
  QmitkWindowsHotkeyHandler& operator=(const QmitkWindowsHotkeyHandler& assignme);


  QmitkWindowsHotkeyHandlerImpl*      m_Pimpl;
  Hotkey                              m_Hotkey;
  volatile bool                       m_ShouldQuit;
};


#endif // QmitkWindowsHotkeyHandler_h
