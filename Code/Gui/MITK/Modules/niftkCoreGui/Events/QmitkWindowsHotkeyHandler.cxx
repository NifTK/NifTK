/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkWindowsHotkeyHandler.h"
#include <sstream>
#include <mitkLogMacros.h>


//-----------------------------------------------------------------------------
struct QmitkWindowsHotkeyHandlerImpl
{
  ATOM      m_HotkeyAtom;
  DWORD     m_HandlerThreadId;

  QmitkWindowsHotkeyHandlerImpl()
    : m_HotkeyAtom(0)
    , m_HandlerThreadId(0)
  {
  }
};


//-----------------------------------------------------------------------------
QmitkWindowsHotkeyHandler::QmitkWindowsHotkeyHandler(Hotkey hk)
  : m_Pimpl(new QmitkWindowsHotkeyHandlerImpl)
  , m_Hotkey(hk)
  , m_ShouldQuit(false)
{
  // not much to do here.
  // registration needs to happen on our own thread.

  // BEWARE: do not moveToThread() this object to itself!
  // because this thread does not run a qt message loop, signals would never
  // get delivered. instead, caller's thread is responsible for signal delivery.

  start();
}


//-----------------------------------------------------------------------------
QmitkWindowsHotkeyHandler::~QmitkWindowsHotkeyHandler()
{
  m_ShouldQuit = true;

  if (m_Pimpl != 0)
  {
    // send a message too, otherwise we could be waiting indefinitely here.
    // why not send WM_QUIT? because i dont know for how long the thread id is valid for.
    // that depends on internals of Qt. so i could be sending a quit message to the wrong
    // thread, terminating an arbitrary application on the users machine.
    PostThreadMessage(m_Pimpl->m_HandlerThreadId, WM_NULL, 0, 0);
    this->wait();

    if (m_Pimpl->m_HotkeyAtom != 0)
    {
      UnregisterHotKey(0, m_Pimpl->m_HotkeyAtom);
      GlobalDeleteAtom(m_Pimpl->m_HotkeyAtom);
    }
    delete m_Pimpl;
  }
}


//-----------------------------------------------------------------------------
void QmitkWindowsHotkeyHandler::run()
{
  assert(QThread::currentThread() == this);

  // need to post quit message.
  // qt has a static thread-id method for the currently running thread,
  // but none for another instance.
  m_Pimpl->m_HandlerThreadId = GetCurrentThreadId();

  // come up with an almost unique name for the hotkey.
  std::ostringstream    name;
  name << "niftk." << m_Hotkey;

  m_Pimpl->m_HotkeyAtom = GlobalAddAtom(name.str().c_str());
  if (m_Pimpl->m_HotkeyAtom == 0)
  {
    MITK_ERROR << "Cannot register hotkey name";
    return;
  }

  int   modifiers = m_Hotkey >> 16;
  int   keycode   = m_Hotkey & 0xFFFF;

  BOOL  ok = RegisterHotKey(0, m_Pimpl->m_HotkeyAtom, modifiers, keycode);
  if (!ok)
  {
    MITK_ERROR << "Cannot register hotkey";

    GlobalDeleteAtom(m_Pimpl->m_HotkeyAtom);
    delete m_Pimpl;
    m_Pimpl = 0;

    return;
  }


  // dont run the qt message loop. we need low-level access.
  while (!m_ShouldQuit)
  {
    MSG     msg;
    if (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE) != 0)
    {
      switch (msg.message)
      {
        case WM_HOTKEY:
          if ((LOWORD(msg.lParam) == modifiers) && (HIWORD(msg.lParam) == keycode))
          {
            MITK_INFO << "Hotkey pressed";
            emit HotkeyPressed(this, m_Hotkey);
          }
          break;

        // is send simply to get this loop to check the value of m_ShouldQuit.
        // but take no other action!
        case WM_NULL:
          break;

        case WM_QUIT:
          m_ShouldQuit = true;
          break;

        default:
          TranslateMessage(&msg);
          DispatchMessageA(&msg);
      }
    }
    else
      WaitMessage();
  }

  // normal cleanup happens in destructor.
}

