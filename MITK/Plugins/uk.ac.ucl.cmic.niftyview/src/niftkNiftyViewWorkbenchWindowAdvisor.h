/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyViewWorkbenchWindowAdvisor_h
#define niftkNiftyViewWorkbenchWindowAdvisor_h

#include <uk_ac_ucl_cmic_niftyview_Export.h>
#include <niftkBaseWorkbenchWindowAdvisor.h>


namespace niftk
{

/**
 * \class NiftyViewWorkbenchWindowAdvisor
 * \brief Advisor class to set up NiftyView windows on startup.
 * \ingroup uk_ac_ucl_cmic_niftyview
 * \sa niftk::HelpAboutDialog
 */
class NIFTYVIEW_EXPORT NiftyViewWorkbenchWindowAdvisor : public BaseWorkbenchWindowAdvisor
{
  Q_OBJECT

public:

  NiftyViewWorkbenchWindowAdvisor(berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer);

};

}

#endif
