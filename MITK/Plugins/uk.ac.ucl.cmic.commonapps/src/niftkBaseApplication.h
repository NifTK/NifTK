/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkBaseApplication_h
#define niftkBaseApplication_h

#include <uk_ac_ucl_cmic_commonapps_Export.h>

#include <berryIApplication.h>
#include <berryWorkbenchAdvisor.h>


namespace niftk
{

/**
 * \class BaseApplication
 * \brief Abstract plugin class to start up an application.
 * \ingroup uk_ac_ucl_cmic_commonapps
 */
class COMMONAPPS_EXPORT BaseApplication : public QObject, public berry::IApplication
{
  Q_OBJECT
  Q_INTERFACES(berry::IApplication)

public:

  BaseApplication();
  BaseApplication(const BaseApplication& other);

  QVariant Start(berry::IApplicationContext* context) override;
  void Stop() override;

protected:

  /// \brief Derived classes override this to provide a workbench advisor.
  virtual berry::WorkbenchAdvisor* GetWorkbenchAdvisor() = 0;
};

}

#endif
