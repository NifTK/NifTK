/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyMITKApplication_h
#define niftkNiftyMITKApplication_h

#include <uk_ac_ucl_cmic_niftymitk_Export.h>
#include <niftkBaseApplication.h>


namespace niftk
{

/**
 * \class NiftyMITKApplication
 * \brief Plugin class to start up the NiftyMITK application.
 * \ingroup uk_ac_ucl_cmic_niftyview
 */
class NIFTYMITK_EXPORT NiftyMITKApplication : public BaseApplication
{
  Q_OBJECT

public:

  NiftyMITKApplication();
  NiftyMITKApplication(const NiftyMITKApplication& other);

protected:

  /// \brief Derived classes override this to provide a workbench advisor.
  virtual berry::WorkbenchAdvisor* GetWorkbenchAdvisor() override;

};

}

#endif
