/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkNiftyIGIApplication_h
#define QmitkNiftyIGIApplication_h

#include <uk_ac_ucl_cmic_gui_qt_niftyigi_Export.h>
#include <QmitkBaseApplication.h>

/**
 * \class QmitkNiftyIGIApplication
 * \brief Plugin class to start up the NiftyIGI application.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyigi
 */
class CMIC_QT_NIFTYIGIAPP QmitkNiftyIGIApplication : public QmitkBaseApplication
{
  Q_OBJECT
  Q_INTERFACES(berry::IApplication)

public:

  QmitkNiftyIGIApplication();
  QmitkNiftyIGIApplication(const QmitkNiftyIGIApplication& other);

protected:

  /// \brief Derived classes override this to provide a workbench advisor.
  virtual berry::WorkbenchAdvisor* GetWorkbenchAdvisor();

};

#endif /*QMITKNIFTYIGIAPPLICATION_H_*/
