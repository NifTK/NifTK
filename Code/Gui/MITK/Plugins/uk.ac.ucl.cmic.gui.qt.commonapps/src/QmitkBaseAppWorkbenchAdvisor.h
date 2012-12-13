/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKBASEAPPWORKBENCHADVISOR_H_
#define QMITKBASEAPPWORKBENCHADVISOR_H_

#ifdef __MINGW32__
// We need to inlclude winbase.h here in order to declare
// atomic intrinsics like InterlockedIncrement correctly.
// Otherwhise, they would be declared wrong within qatomic_windows.h .
#include <windows.h>
#endif

#include <uk_ac_ucl_cmic_gui_qt_commonapps_Export.h>
#include <berryQtWorkbenchAdvisor.h>
#include <berryIWorkbenchConfigurer.h>

class QmitkBaseWorkbenchWindowAdvisor;

/**
 * \class QmitkBaseAppWorkbenchAdvisor
 * \brief Abstract advisor class to set up an initial workbench.
 * \ingroup uk_ac_ucl_cmic_gui_qt_commonapps
 */
class CMIC_QT_COMMONAPPS QmitkBaseAppWorkbenchAdvisor: public berry::QtWorkbenchAdvisor
{
public:

  virtual void Initialize(berry::IWorkbenchConfigurer::Pointer configurer);

  /**
   * \brief Called by framework to create the WorkbenchWindowAdvisor,
   * and derived classes should instead override CreateQmitkBaseWorkbenchWindowAdvisor.
   */
  virtual berry::WorkbenchWindowAdvisor* CreateWorkbenchWindowAdvisor(
      berry::IWorkbenchWindowConfigurer::Pointer configurer);

  /**
   * \brief Derived classes should provide a default perspective id
   * corresponding to a valid perspective defined in their application.
   */
  virtual std::string GetInitialWindowPerspectiveId() = 0;

protected:

  /**
   * \brief Derived classes should provide a window Icon resource path
   * corresponding to a valid icon file, described using a Qt resource location.
   */
  virtual std::string GetWindowIconResourcePath() const = 0;

  /**
   * \brief Actually creates the derived WorkbenchWindowAdvisor.
   */
  virtual QmitkBaseWorkbenchWindowAdvisor* CreateQmitkBaseWorkbenchWindowAdvisor(
      berry::IWorkbenchWindowConfigurer::Pointer configurer);

};

#endif /*QMITKBASEAPPWORKBENCHADVISOR_H_*/
