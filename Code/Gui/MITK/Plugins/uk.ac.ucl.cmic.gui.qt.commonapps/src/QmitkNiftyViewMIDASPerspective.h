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


#ifndef QMITKNIFTYVIEWMIDASPERSPECTIVE_H_
#define QMITKNIFTYVIEWMIDASPERSPECTIVE_H_

#include "mitkQtCommonAppsAppDll.h"

#include <berryIPerspectiveFactory.h>

/**
 * \class QmitkNiftyViewMIDASPerspective
 * \brief Perspective to arrange widgets as would be suitable for MIDAS applications,
 * where the standard view has up to 5x5 independent windows.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyview_internal
 *
 * Note: We have to load at least one view component, to get an editor created.
 */
class CMIC_QT_COMMONAPPS QmitkNiftyViewMIDASPerspective : public QObject, public berry::IPerspectiveFactory
{
  Q_OBJECT
  Q_INTERFACES(berry::IPerspectiveFactory)
  
public:

  QmitkNiftyViewMIDASPerspective();
  QmitkNiftyViewMIDASPerspective(const QmitkNiftyViewMIDASPerspective& other);
  
  void CreateInitialLayout(berry::IPageLayout::Pointer layout);

};

#endif /* QMITKNIFTYVIEWMIDASPERSPECTIVE_H_ */
