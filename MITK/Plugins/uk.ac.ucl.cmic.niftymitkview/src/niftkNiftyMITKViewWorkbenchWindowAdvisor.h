/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyMITKViewWorkbenchWindowAdvisor_h
#define niftkNiftyMITKViewWorkbenchWindowAdvisor_h

#include <uk_ac_ucl_cmic_niftymitkview_Export.h>
#include <niftkBaseWorkbenchWindowAdvisor.h>


namespace niftk
{

/**
 * \class NiftyMITKViewWorkbenchWindowAdvisor
 * \brief Advisor class to set up NiftyMITKView windows on startup.
 * \ingroup uk_ac_ucl_cmic_niftyview
 * \sa niftk::HelpAboutDialog
 */
class NIFTYMITKVIEW_EXPORT NiftyMITKViewWorkbenchWindowAdvisor : public BaseWorkbenchWindowAdvisor
{
  Q_OBJECT

public:

  NiftyMITKViewWorkbenchWindowAdvisor(berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer);

  /**
   * \brief We override BaseWorkbenchWindowAdvisor::PostWindowCreate
   * to additionally provide an option to force the MITK display open
   * with an environment variable called NIFTK_MITK_DISPLAY=ON.
   */
  virtual void PostWindowCreate() override;
};

}

#endif
