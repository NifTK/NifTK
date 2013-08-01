/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkBaseApplication_h
#define QmitkBaseApplication_h

#include <uk_ac_ucl_cmic_gui_qt_commonapps_Export.h>
#include <berryIApplication.h>
#include <berryWorkbenchAdvisor.h>

/**
 * \class QmitkBaseApplication
 * \brief Abstract plugin class to start up an application.
 * \ingroup uk_ac_ucl_cmic_gui_qt_commonapps_internal
 */
class CMIC_QT_COMMONAPPS QmitkBaseApplication : public QObject, public berry::IApplication
{
  Q_OBJECT
  Q_INTERFACES(berry::IApplication)

public:

  QmitkBaseApplication();
  QmitkBaseApplication(const QmitkBaseApplication& other);

  int Start();
  void Stop();

protected:

  /// \brief Derived classes override this to provide a workbench advisor.
  virtual berry::WorkbenchAdvisor* GetWorkbenchAdvisor() = 0;
};

#endif /*QMITKBASEAPPLICATION_H_*/
