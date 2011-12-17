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


#ifndef QMITKNIFTYVIEWCMICPERSPECTIVE_H_
#define QMITKNIFTYVIEWCMICPERSPECTIVE_H_

#include <berryIPerspectiveFactory.h>

/**
 * \class QmitkNiftyViewCMICPerspective
 * \brief Perspective to arrange widgets as would be suitable for CMIC applications that
 * typically want an FSL/MITK style orthoviewer as the default look.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyview_internal
 *
 * Note: We have to load at least one view component, to get an editor created.
 */
class QmitkNiftyViewCMICPerspective : public QObject, public berry::IPerspectiveFactory
{
  Q_OBJECT
  Q_INTERFACES(berry::IPerspectiveFactory)
  
public:

  QmitkNiftyViewCMICPerspective();
  QmitkNiftyViewCMICPerspective(const QmitkNiftyViewCMICPerspective& other);
  
  void CreateInitialLayout(berry::IPageLayout::Pointer layout);

};

#endif /* QMITKNIFTYVIEWDEFAULTPERSPECTIVE_H_ */
