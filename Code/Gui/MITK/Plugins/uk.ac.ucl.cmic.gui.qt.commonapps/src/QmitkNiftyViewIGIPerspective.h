/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKNIFTYVIEWIGIPERSPECTIVE_H_
#define QMITKNIFTYVIEWIGIPERSPECTIVE_H_

#include <uk_ac_ucl_cmic_gui_qt_commonapps_Export.h>
#include <berryIPerspectiveFactory.h>

/**
 * \class QmitkNiftyViewIGIPerspective
 * \brief Perspective to arrange widgets as would be suitable for CMIC IGI applications.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyview_internal
 *
 * Note: We have to load at least one view component, to get an editor created.
 */
class CMIC_QT_COMMONAPPS QmitkNiftyViewIGIPerspective : public QObject, public berry::IPerspectiveFactory
{
  Q_OBJECT
  Q_INTERFACES(berry::IPerspectiveFactory)
  
public:

  QmitkNiftyViewIGIPerspective();
  QmitkNiftyViewIGIPerspective(const QmitkNiftyViewIGIPerspective& other);
  
  void CreateInitialLayout(berry::IPageLayout::Pointer layout);

};

#endif /* QMITKNIFTYVIEWIGIPERSPECTIVE_H_ */
