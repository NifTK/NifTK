/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyViewAppWorkbenchAdvisor_h
#define niftkNiftyViewAppWorkbenchAdvisor_h

#ifdef __MINGW32__
// We need to include winbase.h here in order to declare
// atomic intrinsics like InterlockedIncrement correctly.
// Otherwhise, they would be declared wrong within qatomic_windows.h .
#include <windows.h>
#endif

#include <uk_ac_ucl_cmic_niftyview_Export.h>
#include <niftkBaseWorkbenchAdvisor.h>


namespace niftk
{

/**
 * \class NiftyViewAppWorkbenchAdvisor
 * \brief Advisor class to set up the initial NiftyView workbench.
 * \ingroup uk_ac_ucl_cmic_niftyview
 */
class NIFTYVIEW_EXPORT NiftyViewAppWorkbenchAdvisor: public BaseWorkbenchAdvisor
{
public:

  /// \brief Returns uk.ac.ucl.cmic.niftyview.midasperspective which should match that in plugin.xml.
  virtual QString GetInitialWindowPerspectiveId() override;

  /// \brief Gets the resource name of the window icon.
  virtual QString GetWindowIconResourcePath() const override;

protected:

  /**
   * \brief Actually creates the derived WorkbenchWindowAdvisor.
   */
  virtual BaseWorkbenchWindowAdvisor* CreateBaseWorkbenchWindowAdvisor(
      berry::IWorkbenchWindowConfigurer::Pointer configurer) override;

};

}

#endif
