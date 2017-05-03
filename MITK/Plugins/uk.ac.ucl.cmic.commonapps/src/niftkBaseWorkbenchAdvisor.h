/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkBaseWorkbenchAdvisor_h
#define niftkBaseWorkbenchAdvisor_h

#ifdef __MINGW32__
// We need to inlclude winbase.h here in order to declare
// atomic intrinsics like InterlockedIncrement correctly.
// Otherwhise, they would be declared wrong within qatomic_windows.h .
#include <windows.h>
#endif

#include <uk_ac_ucl_cmic_commonapps_Export.h>
#include <berryQtWorkbenchAdvisor.h>
#include <berryIWorkbenchConfigurer.h>

namespace mitk
{
class DataStorage;
}


namespace niftk
{

class BaseWorkbenchWindowAdvisor;

/**
 * \class BaseWorkbenchAdvisor
 * \brief Abstract advisor class to set up an initial workbench.
 * \ingroup uk_ac_ucl_cmic_common
 */
class COMMONAPPS_EXPORT BaseWorkbenchAdvisor: public berry::QtWorkbenchAdvisor
{
public:

  virtual void Initialize(berry::IWorkbenchConfigurer::Pointer configurer) override;

  /**
   * \brief Called by framework to create the WorkbenchWindowAdvisor,
   * and derived classes should instead override CreateBaseWorkbenchWindowAdvisor.
   */
  virtual berry::WorkbenchWindowAdvisor* CreateWorkbenchWindowAdvisor(
      berry::IWorkbenchWindowConfigurer::Pointer configurer) override;

  virtual void PostStartup() override;

  /**
   * Overriden from berry::WorkbenchAdvisor so that we can pop up a dialog box
   * asking the user whether they really want to close the application.
   * @returns true if closing is ok, false otherwise.
   */
  virtual bool PreShutdown() override;

  void SetPerspective(const QString& perspectiveLabel);

  void ResetPerspective();

protected:

  /**
   * \brief Derived classes should provide a window Icon resource path
   * corresponding to a valid icon file, described using a Qt resource location.
   */
  virtual QString GetWindowIconResourcePath() const = 0;

  /**
   * \brief Actually creates the derived WorkbenchWindowAdvisor.
   */
  virtual BaseWorkbenchWindowAdvisor* CreateBaseWorkbenchWindowAdvisor(
      berry::IWorkbenchWindowConfigurer::Pointer configurer);

  mitk::DataStorage* GetDataStorage();

};

}

#endif
