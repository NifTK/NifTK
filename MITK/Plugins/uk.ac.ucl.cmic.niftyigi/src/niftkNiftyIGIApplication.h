/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyIGIApplication_h
#define niftkNiftyIGIApplication_h

#include <uk_ac_ucl_cmic_niftyigi_Export.h>
#include <niftkBaseApplication.h>


namespace niftk
{

/**
 * \class NiftyIGIApplication
 * \brief Plugin class to start up the NiftyIGI application.
 * \ingroup uk_ac_ucl_cmic_niftyigi
 */
class NIFTYIGI_EXPORT NiftyIGIApplication : public BaseApplication
{
  Q_OBJECT
  Q_INTERFACES(berry::IApplication)

public:

  NiftyIGIApplication();
  NiftyIGIApplication(const NiftyIGIApplication& other);

protected:

  /// \brief Derived classes override this to provide a workbench advisor.
  virtual berry::WorkbenchAdvisor* GetWorkbenchAdvisor() override;

};

}

#endif
