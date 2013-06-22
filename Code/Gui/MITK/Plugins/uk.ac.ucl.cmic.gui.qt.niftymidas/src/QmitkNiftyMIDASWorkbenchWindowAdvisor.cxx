/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkNiftyMIDASWorkbenchWindowAdvisor.h"

#include <QMainWindow>
#include <QStatusBar>

#include <QmitkMemoryUsageIndicatorView.h>

//-----------------------------------------------------------------------------
QmitkNiftyMIDASWorkbenchWindowAdvisor::QmitkNiftyMIDASWorkbenchWindowAdvisor(
    berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
: QmitkBaseWorkbenchWindowAdvisor(wbAdvisor, configurer)
{
}


//-----------------------------------------------------------------------------
void QmitkNiftyMIDASWorkbenchWindowAdvisor::PostWindowCreate()
{
  QmitkBaseWorkbenchWindowAdvisor::PostWindowCreate();
  this->CheckIfLoadingMITKDisplay();

  // very bad hack...
  berry::IWorkbenchWindow::Pointer window = this->GetWindowConfigurer()->GetWindow();
  QMainWindow* mainWindow = static_cast<QMainWindow*>(window->GetShell()->GetControl());

  // Here we turn off the word wrap property of the memory usage label.
  // If the property is on then it can cause that the height of the status bar increases
  // and the editors (displays) "jump up".
  if (QStatusBar* statusBar = mainWindow->statusBar())
  {
    if (QmitkMemoryUsageIndicatorView* memoryUsageIndicator = statusBar->findChild<QmitkMemoryUsageIndicatorView*>())
    {
      if (QLabel* memoryUsageLabel = memoryUsageIndicator->findChild<QLabel*>("m_Label"))
      {
        memoryUsageLabel->setWordWrap(false);
      }
    }
  }
}
