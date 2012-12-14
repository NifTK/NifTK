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
#ifndef QMITKNIFTYIGIWORKBENCHWINDOWADVISOR_H_
#define QMITKNIFTYIGIWORKBENCHWINDOWADVISOR_H_

#include "mitkQtNiftyIGIAppDll.h"
#include "QmitkBaseWorkbenchWindowAdvisor.h"

/**
 * \class QmitkNiftyIGIWorkbenchWindowAdvisor
 * \brief Advisor class to set up NiftyIGI windows on startup.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyigi
 * \sa QmitkHelpAboutDialog
 */
class CMIC_QT_NIFTYIGIAPP QmitkNiftyIGIWorkbenchWindowAdvisor : public QmitkBaseWorkbenchWindowAdvisor
{
  Q_OBJECT

public:

  QmitkNiftyIGIWorkbenchWindowAdvisor(berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer);
};

#endif /*QMITKNIFTYIGIWORKBENCHWINDOWADVISOR_H_*/

