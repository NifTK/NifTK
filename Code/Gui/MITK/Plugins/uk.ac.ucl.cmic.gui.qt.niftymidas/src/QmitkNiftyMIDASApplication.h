/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKNIFTYMIDASAPPLICATION_H_
#define QMITKNIFTYMIDASAPPLICATION_H_

#include <uk_ac_ucl_cmic_gui_qt_niftymidas_Export.h>
#include "QmitkBaseApplication.h"

/**
 * \class QmitkNiftyMIDASApplication
 * \brief Plugin class to start up the NiftyMIDAS application.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftymidas
 */
class CMIC_QT_NIFTYMIDASAPP QmitkNiftyMIDASApplication : public QmitkBaseApplication
{
  Q_OBJECT
  Q_INTERFACES(berry::IApplication)

public:

  QmitkNiftyMIDASApplication();
  QmitkNiftyMIDASApplication(const QmitkNiftyMIDASApplication& other);

protected:

  /// \brief Derived classes override this to provide a workbench advisor.
  virtual berry::WorkbenchAdvisor* GetWorkbenchAdvisor();

};

#endif /*QMITKNIFTYMIDASAPPLICATION_H_*/
