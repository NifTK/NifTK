/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkCommonAppsMinimalPerspective_h
#define QmitkCommonAppsMinimalPerspective_h

#include <uk_ac_ucl_cmic_gui_qt_commonapps_Export.h>
#include <berryIPerspectiveFactory.h>

/**
 * \class QmitkCommonAppsMinimalPerspective
 * \brief Default Perspective, called 'Minimal' to discourage incrementally adding to it.  
 * \ingroup uk_ac_ucl_cmic_gui_qt_commonapps_internal
 *
 * Note: We have to load at least one view component, to get an editor created.
 */
class CMIC_QT_COMMONAPPS QmitkCommonAppsMinimalPerspective : public QObject, public berry::IPerspectiveFactory
{
  Q_OBJECT
  Q_INTERFACES(berry::IPerspectiveFactory)
  
public:

  QmitkCommonAppsMinimalPerspective();
  QmitkCommonAppsMinimalPerspective(const QmitkCommonAppsMinimalPerspective& other);
  
  void CreateInitialLayout(berry::IPageLayout::Pointer layout);

};

#endif 
