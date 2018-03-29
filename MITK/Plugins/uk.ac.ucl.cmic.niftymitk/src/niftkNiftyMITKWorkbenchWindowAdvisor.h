/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyMITKWorkbenchWindowAdvisor_h
#define niftkNiftyMITKWorkbenchWindowAdvisor_h

#include <uk_ac_ucl_cmic_niftymitk_Export.h>
#include <niftkBaseWorkbenchWindowAdvisor.h>


namespace niftk
{

/**
 * \class NiftyMITKWorkbenchWindowAdvisor
 * \brief Advisor class to set up NiftyMITK windows on startup.
 * \ingroup uk_ac_ucl_cmic_niftyview
 * \sa niftk::HelpAboutDialog
 */
class NIFTYMITK_EXPORT NiftyMITKWorkbenchWindowAdvisor : public BaseWorkbenchWindowAdvisor
{
  Q_OBJECT

public:

  NiftyMITKWorkbenchWindowAdvisor(berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer);

};

}

#endif
