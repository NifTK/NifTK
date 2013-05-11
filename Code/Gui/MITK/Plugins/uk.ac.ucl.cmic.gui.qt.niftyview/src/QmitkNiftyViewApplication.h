/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKNIFTYVIEWAPPLICATION_H_
#define QMITKNIFTYVIEWAPPLICATION_H_

#include <uk_ac_ucl_cmic_gui_qt_niftyview_Export.h>
#include <QmitkBaseApplication.h>

/**
 * \class QmitkNiftyViewApplication
 * \brief Plugin class to start up the NiftyView application.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyview
 */
class CMIC_QT_NIFTYVIEWAPP QmitkNiftyViewApplication : public QmitkBaseApplication
{
  Q_OBJECT
  Q_INTERFACES(berry::IApplication)

public:

  QmitkNiftyViewApplication();
  QmitkNiftyViewApplication(const QmitkNiftyViewApplication& other);

protected:

  /// \brief Derived classes override this to provide a workbench advisor.
  virtual berry::WorkbenchAdvisor* GetWorkbenchAdvisor();

};

#endif /*QMITKNIFTYVIEWAPPLICATION_H_*/
