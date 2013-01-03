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

#ifndef QMITKBASEAPPLICATION_H_
#define QMITKBASEAPPLICATION_H_

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
