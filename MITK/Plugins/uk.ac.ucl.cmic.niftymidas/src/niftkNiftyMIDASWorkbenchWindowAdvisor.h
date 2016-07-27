/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyMIDASWorkbenchWindowAdvisor_h
#define niftkNiftyMIDASWorkbenchWindowAdvisor_h

#include <uk_ac_ucl_cmic_niftymidas_Export.h>

#include <niftkBaseWorkbenchWindowAdvisor.h>


namespace niftk
{

/**
 * \class NiftyMIDASWorkbenchWindowAdvisor
 * \brief Advisor class to set up NiftyMIDAS windows on startup.
 * \ingroup uk_ac_ucl_cmic_niftymidas
 * \sa niftk::HelpAboutDialog
 */
class NIFTYMIDAS_EXPORT NiftyMIDASWorkbenchWindowAdvisor : public BaseWorkbenchWindowAdvisor
{
  Q_OBJECT

public:

  NiftyMIDASWorkbenchWindowAdvisor(berry::WorkbenchAdvisor* wbAdvisor,
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
