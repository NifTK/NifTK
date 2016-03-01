/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkNiftyMIDASWorkbenchWindowAdvisor_h
#define QmitkNiftyMIDASWorkbenchWindowAdvisor_h

#include <uk_ac_ucl_cmic_gui_qt_niftymidas_Export.h>
#include <QmitkBaseWorkbenchWindowAdvisor.h>

/**
 * \class QmitkNiftyMIDASWorkbenchWindowAdvisor
 * \brief Advisor class to set up NiftyMIDAS windows on startup.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftymidas
 * \sa QmitkHelpAboutDialog
 */
class CMIC_QT_NIFTYMIDASAPP QmitkNiftyMIDASWorkbenchWindowAdvisor : public QmitkBaseWorkbenchWindowAdvisor
{
  Q_OBJECT

public:

  QmitkNiftyMIDASWorkbenchWindowAdvisor(berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer);

  /**
   * \brief We override QmitkBaseWorkbenchWindowAdvisor::PostWindowCreate
   * to additionally provide an option to force the MITK display open
   * with an environment variable called NIFTK_MITK_DISPLAY=ON.
   */
  virtual void PostWindowCreate();
};

#endif /*QMITKNIFTYMIDASWORKBENCHWINDOWADVISOR_H_*/

