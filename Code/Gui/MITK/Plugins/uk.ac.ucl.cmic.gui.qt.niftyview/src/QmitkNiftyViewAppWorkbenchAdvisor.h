/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKNIFTYVIEWAPPWORKBENCHADVISOR_H_
#define QMITKNIFTYVIEWAPPWORKBENCHADVISOR_H_

#ifdef __MINGW32__
// We need to inlclude winbase.h here in order to declare
// atomic intrinsics like InterlockedIncrement correctly.
// Otherwhise, they would be declared wrong within qatomic_windows.h .
#include <windows.h>
#endif

#include "mitkQtNiftyViewAppDll.h"
#include "QmitkBaseAppWorkbenchAdvisor.h"

/**
 * \class QmitkNiftyViewAppWorkbenchAdvisor
 * \brief Advisor class to set up the initial NiftyView workbench.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyview
 */
class CMIC_QT_NIFTYVIEWAPP QmitkNiftyViewAppWorkbenchAdvisor: public QmitkBaseAppWorkbenchAdvisor
{
public:

  /// \brief Returns uk.ac.ucl.cmic.gui.qt.niftyview.midasperspective which should match that in plugin.xml.
  virtual std::string GetInitialWindowPerspectiveId();

  /// \brief Gets the resource name of the window icon.
  virtual std::string GetWindowIconResourcePath() const;
};

#endif /*QMITKEXTAPPWORKBENCHADVISOR_H_*/
