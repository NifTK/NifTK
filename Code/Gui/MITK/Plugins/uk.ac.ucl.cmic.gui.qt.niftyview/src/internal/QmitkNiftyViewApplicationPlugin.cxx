/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkNiftyViewApplicationPlugin.h"
#include <QmitkCommonAppsMinimalPerspective.h>
#include <QmitkCommonAppsIGIPerspective.h>
#include <QmitkMIDASSegmentationPerspective.h>
#include <QmitkNiftyViewApplicationPreferencePage.h>
#include <QmitkCommonAppsApplicationPreferencePage.h>
#include "../QmitkNiftyViewApplication.h"

//-----------------------------------------------------------------------------
QmitkNiftyViewApplicationPlugin::QmitkNiftyViewApplicationPlugin()
{
}


//-----------------------------------------------------------------------------
QmitkNiftyViewApplicationPlugin::~QmitkNiftyViewApplicationPlugin()
{
}


//-----------------------------------------------------------------------------
QString QmitkNiftyViewApplicationPlugin::GetHelpHomePageURL() const
{
  return QString("qthelp://uk.ac.ucl.cmic.gui.qt.niftyview/bundle/uk_ac_ucl_cmic_gui_qt_niftyview_intro.html");
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPlugin::start(ctkPluginContext* context)
{
  /// Note:
  /// This function has to be redefined so that the superclass
  /// implementation does not run again. The overridden function 
  /// has been executed when the commonapps plugin has been loaded.

  this->SetPluginContext(context);
  
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewApplication, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkCommonAppsMinimalPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkCommonAppsIGIPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkCommonAppsApplicationPreferencePage, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewApplicationPreferencePage, context);

  this->RegisterHelpSystem();
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPlugin::stop(ctkPluginContext* context)
{
  /// Note:
  /// This function has to be redefined so that the superclass
  /// implementation does not run again. The overridden function 
  /// will be executed when the commonapps plugin gets unloaded.
}


//-----------------------------------------------------------------------------
Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_gui_qt_niftyview, QmitkNiftyViewApplicationPlugin)
