/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKNIFTYIGIWORKBENCHWINDOWADVISOR_H_
#define QMITKNIFTYIGIWORKBENCHWINDOWADVISOR_H_

#include <uk_ac_ucl_cmic_gui_qt_niftyigi_Export.h>
#include "QmitkBaseWorkbenchWindowAdvisor.h"

/**
 * \class QmitkNiftyIGIWorkbenchWindowAdvisor
 * \brief Advisor class to set up NiftyIGI windows on startup.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyigi
 * \sa QmitkHelpAboutDialog
 */
class CMIC_QT_NIFTYIGIAPP QmitkNiftyIGIWorkbenchWindowAdvisor : public QmitkBaseWorkbenchWindowAdvisor
{
  Q_OBJECT

public:

  QmitkNiftyIGIWorkbenchWindowAdvisor(berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer);
};

#endif /*QMITKNIFTYIGIWORKBENCHWINDOWADVISOR_H_*/

