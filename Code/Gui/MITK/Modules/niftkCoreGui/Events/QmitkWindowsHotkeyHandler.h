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


struct QmitkWindowsHotkeyHandlerImpl;

class NIFTKCOREGUI_EXPORT QmitkWindowsHotkeyHandler : public QThread
{
  Q_OBJECT

public:
  enum Hotkey
  {
    CTRL_ALT_F5     = 0x00030074    // VK_F5 | ((MOD_ALT | MOD_CONTROL) << 16)
  };

  QmitkWindowsHotkeyHandler(Hotkey hk);
  virtual ~QmitkWindowsHotkeyHandler();


signals:
  /**
   * Beware: you really should connect this as QueuedConnection!
   */
  void HotkeyPressed(QmitkWindowsHotkeyHandler* sender, int hotkey);


protected:
  virtual void run();


private:
  QmitkWindowsHotkeyHandler(const QmitkWindowsHotkeyHandler& copyme);
  QmitkWindowsHotkeyHandler& operator=(const QmitkWindowsHotkeyHandler& assignme);


  QmitkWindowsHotkeyHandlerImpl*      m_Pimpl;
  Hotkey                              m_Hotkey;
  volatile bool                       m_ShouldQuit;
};


#endif // QmitkWindowsHotkeyHandler_h
