/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyViewApplication_h
#define niftkNiftyViewApplication_h

#include <uk_ac_ucl_cmic_niftyview_Export.h>
#include <niftkBaseApplication.h>


namespace niftk
{

/**
 * \class NiftyViewApplication
 * \brief Plugin class to start up the NiftyView application.
 * \ingroup uk_ac_ucl_cmic_niftyview
 */
class NIFTYVIEW_EXPORT NiftyViewApplication : public BaseApplication
{
  Q_OBJECT

public:

  NiftyViewApplication();
  NiftyViewApplication(const NiftyViewApplication& other);

protected:

  /// \brief Derived classes override this to provide a workbench advisor.
  virtual berry::WorkbenchAdvisor* GetWorkbenchAdvisor() override;

};

}

#endif
