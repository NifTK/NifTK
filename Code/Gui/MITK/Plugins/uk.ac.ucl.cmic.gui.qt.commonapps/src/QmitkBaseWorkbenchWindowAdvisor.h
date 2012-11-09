/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-20 14:25:53 +0000 (Sun, 20 Nov 2011) $
 Revision          : $Revision: 7818 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKBASEWORKBENCHWINDOWADVISOR_H_
#define QMITKBASEWORKBENCHWINDOWADVISOR_H_

#include "mitkQtCommonAppsAppDll.h"
#include "QmitkExtWorkbenchWindowAdvisor.h"

/**
 * \class QmitkBaseViewWorkbenchWindowAdvisor
 * \brief Abstract advisor class to set up application windows on startup.
 * \ingroup uk_ac_ucl_cmic_gui_qt_commonapps
 * \sa QmitkHelpAboutDialog
 */
class CMIC_QT_COMMONAPPS QmitkBaseWorkbenchWindowAdvisor : public QmitkExtWorkbenchWindowAdvisor
{
  Q_OBJECT

public:

  QmitkBaseWorkbenchWindowAdvisor(berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer);

  /// \brief We override the base class PreWindowOpen to specifically set
  /// QmitkExtWorkbenchWindowAdvisor::showVersionInfo and
  /// QmitkExtWorkbenchWindowAdvisor::showMitkVersionInfo to false.
  virtual void PreWindowOpen();

  /// \brief We override the base class PostWindowCreate to customise
  /// the About dialog box to call our QmitkHelpAboutDialog, and to remove
  /// the Welcome dialog box.
  virtual void PostWindowCreate();

public slots:

  void OnHelpAbout();

};

#endif /*QMITKBASEWORKBENCHWINDOWADVISOR_H_*/

