/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyIGIWorkbenchWindowAdvisor_h
#define niftkNiftyIGIWorkbenchWindowAdvisor_h

#include <uk_ac_ucl_cmic_niftyigi_Export.h>
#include <niftkBaseWorkbenchWindowAdvisor.h>


namespace niftk
{

/**
 * \class NiftyIGIWorkbenchWindowAdvisor
 * \brief Advisor class to set up NiftyIGI windows on startup.
 * \ingroup uk_ac_ucl_cmic_niftyigi
 * \sa niftk::HelpAboutDialog
 */
class NIFTYIGI_EXPORT NiftyIGIWorkbenchWindowAdvisor : public BaseWorkbenchWindowAdvisor
{
  Q_OBJECT

public:

  NiftyIGIWorkbenchWindowAdvisor(berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer);
};

}

#endif
