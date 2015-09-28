/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkNiftyIGIApplicationPlugin.h"
#include <QmitkCommonAppsIGIPerspective.h>
#include "../QmitkNiftyIGIApplication.h"
#include <QmitkNiftyViewApplicationPreferencePage.h>
#include <QmitkCommonAppsApplicationPreferencePage.h>

//-----------------------------------------------------------------------------
QmitkNiftyIGIApplicationPlugin::QmitkNiftyIGIApplicationPlugin()
{
}


//-----------------------------------------------------------------------------
QmitkNiftyIGIApplicationPlugin::~QmitkNiftyIGIApplicationPlugin()
{
}


//-----------------------------------------------------------------------------
QString QmitkNiftyIGIApplicationPlugin::GetHelpHomePageURL() const
{
  return QString("qthelp://uk.ac.ucl.cmic.gui.qt.niftyigi/bundle/uk_ac_ucl_cmic_gui_qt_niftyigi_intro.html");
}


//-----------------------------------------------------------------------------
void QmitkNiftyIGIApplicationPlugin::start(ctkPluginContext* context)
{
  /// Note:
  /// This function has to be redefined so that the superclass
  /// implementation does not run again. The overridden function
  /// has been executed when the commonapps plugin has been loaded.

  this->SetPluginContext(context);

  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyIGIApplication, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkCommonAppsIGIPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkCommonAppsApplicationPreferencePage, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewApplicationPreferencePage, context);

  this->RegisterHelpSystem();
}


//-----------------------------------------------------------------------------
void QmitkNiftyIGIApplicationPlugin::stop(ctkPluginContext* context)
{
  /// Note:
  /// This function has to be redefined so that the superclass
  /// implementation does not run again. The overridden function 
  /// will be executed when the commonapps plugin gets unloaded.
}


//-----------------------------------------------------------------------------
Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_gui_qt_niftyigi, QmitkNiftyIGIApplicationPlugin)
