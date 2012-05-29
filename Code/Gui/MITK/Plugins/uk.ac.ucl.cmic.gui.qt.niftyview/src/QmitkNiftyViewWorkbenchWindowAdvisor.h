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

#ifndef QMITKNIFTYVIEWWORKBENCHWINDOWADVISOR_H_
#define QMITKNIFTYVIEWWORKBENCHWINDOWADVISOR_H_

#include "mitkQtNiftyViewAppDll.h"
#include <QObject>
#include <berryWorkbenchAdvisor.h>
#include "QmitkExtWorkbenchWindowAdvisor.h"

/**
 * \class QmitkNiftyViewWorkbenchWindowAdvisor
 * \brief Advisor class to set up NiftyView windows on startup.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyview
 * \sa QmitkHelpAboutDialog
 */
class CMIC_QT_NIFTYVIEWAPP QmitkNiftyViewWorkbenchWindowAdvisor : public QmitkExtWorkbenchWindowAdvisor
{
  Q_OBJECT

public:

  QmitkNiftyViewWorkbenchWindowAdvisor(berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer);

  /// \brief We override the base class PreWindowOpen to specifically set
  /// QmitkExtWorkbenchWindowAdvisor::showVersionInfo and
  /// QmitkExtWorkbenchWindowAdvisor::showMitkVersionInfo to false.
  virtual void PreWindowOpen();

  /// \brief We override the base class PostWindowCreate to specifically
  /// enable image files to be loaded from the command line and to customise
  /// the About dialog box to call our QmitkHelpAboutDialog, and to remove
  /// the Welcome dialog box.
  virtual void PostWindowCreate();

public slots:

  void OnHelpAbout();

};

#endif /*QMITKEXTWORKBENCHADVISOR_H_*/

