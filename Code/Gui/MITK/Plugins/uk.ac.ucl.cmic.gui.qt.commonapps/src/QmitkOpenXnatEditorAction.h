/*===================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center,
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE.txt or http://www.mitk.org for details.

===================================================================*/

#ifndef QMITKOPENXNATEDITORACTION_H_
#define QMITKOPENXNATEDITORACTION_H_

#ifdef __MINGW32__
// We need to inlclude winbase.h here in order to declare
// atomic intrinsics like InterlockedIncrement correctly.
// Otherwhise, they would be declared wrong within qatomic_windows.h .
#include <windows.h>
#endif

#include <QAction>
#include <QIcon>

#include <uk_ac_ucl_cmic_gui_qt_commonapps_Export.h>

#include <berryIWorkbenchWindow.h>
#include <berryIPreferences.h>

class CMIC_QT_COMMONAPPS QmitkOpenXnatEditorAction : public QAction
{
  Q_OBJECT

public:
  QmitkOpenXnatEditorAction(berry::IWorkbenchWindow::Pointer window);
  QmitkOpenXnatEditorAction(const QIcon & icon, berry::IWorkbenchWindow::Pointer window);

protected slots:

  void Run();

private:
  void init ( berry::IWorkbenchWindow::Pointer window );
  berry::IWorkbenchWindow::Pointer m_Window;
  berry::IPreferences::WeakPtr m_GeneralPreferencesNode;
};


#endif /*QMITKOPENXNATEDITORACTION_H_*/
