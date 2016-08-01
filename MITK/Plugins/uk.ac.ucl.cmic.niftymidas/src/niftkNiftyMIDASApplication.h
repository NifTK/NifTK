/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyMIDASApplication_h
#define niftkNiftyMIDASApplication_h

#include <uk_ac_ucl_cmic_niftymidas_Export.h>

#include <niftkBaseApplication.h>


namespace niftk
{

/**
 * \class NiftyMIDASApplication
 * \brief Plugin class to start up the NiftyMIDAS application.
 * \ingroup uk_ac_ucl_cmic_niftymidas
 */
class NIFTYMIDAS_EXPORT NiftyMIDASApplication : public BaseApplication
{
  Q_OBJECT
  Q_INTERFACES(berry::IApplication)

public:

  NiftyMIDASApplication();
  NiftyMIDASApplication(const NiftyMIDASApplication& other);

protected:

  /// \brief Derived classes override this to provide a workbench advisor.
  virtual berry::WorkbenchAdvisor* GetWorkbenchAdvisor();

};

}

#endif
