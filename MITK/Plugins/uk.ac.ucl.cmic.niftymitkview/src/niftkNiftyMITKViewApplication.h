/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyMITKViewApplication_h
#define niftkNiftyMITKViewApplication_h

#include <uk_ac_ucl_cmic_niftymitkview_Export.h>
#include <niftkBaseApplication.h>


namespace niftk
{

/**
 * \class NiftyMITKViewApplication
 * \brief Plugin class to start up the NiftyMITKView application.
 * \ingroup uk_ac_ucl_cmic_niftyview
 */
class NIFTYMITKVIEW_EXPORT NiftyMITKViewApplication : public BaseApplication
{
  Q_OBJECT

public:

  NiftyMITKViewApplication();
  NiftyMITKViewApplication(const NiftyMITKViewApplication& other);

protected:

  /// \brief Derived classes override this to provide a workbench advisor.
  virtual berry::WorkbenchAdvisor* GetWorkbenchAdvisor() override;

};

}

#endif
