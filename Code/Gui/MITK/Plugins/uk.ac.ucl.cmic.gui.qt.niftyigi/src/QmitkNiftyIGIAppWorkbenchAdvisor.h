/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKNIFTYIGIAPPWORKBENCHADVISOR_H_
#define QMITKNIFTYIGIAPPWORKBENCHADVISOR_H_

#ifdef __MINGW32__
// We need to inlclude winbase.h here in order to declare
// atomic intrinsics like InterlockedIncrement correctly.
// Otherwhise, they would be declared wrong within qatomic_windows.h .
#include <windows.h>
#endif

#include <uk_ac_ucl_cmic_gui_qt_niftyigi_Export.h>
#include "QmitkBaseAppWorkbenchAdvisor.h"

/**
 * \class QmitkNiftyIGIAppWorkbenchAdvisor
 * \brief Advisor class to set up the initial NiftyIGI workbench.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyigi
 */
class CMIC_QT_NIFTYIGIAPP QmitkNiftyIGIAppWorkbenchAdvisor: public QmitkBaseAppWorkbenchAdvisor
{
public:

  /// \brief Returns uk.ac.ucl.cmic.gui.qt.niftyview.igiperspective which should match that in plugin.xml.
  virtual std::string GetInitialWindowPerspectiveId();

  /// \brief Gets the resource name of the window icon.
  virtual std::string GetWindowIconResourcePath() const;

protected:

  /**
   * \brief Actually creates the derived WorkbenchWindowAdvisor.
   */
  virtual QmitkBaseWorkbenchWindowAdvisor* CreateQmitkBaseWorkbenchWindowAdvisor(
      berry::IWorkbenchWindowConfigurer::Pointer configurer);

};

#endif /*QMITKNIFTYIGIAPPWORKBENCHADVISOR_H_*/
