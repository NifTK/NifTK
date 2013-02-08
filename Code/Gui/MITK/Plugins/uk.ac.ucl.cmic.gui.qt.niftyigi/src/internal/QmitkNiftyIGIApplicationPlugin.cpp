/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkNiftyIGIApplicationPlugin.h"
#include "QmitkNiftyViewIGIPerspective.h"
#include "../QmitkNiftyIGIApplication.h"

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
  return QString("qthelp://uk.ac.ucl.cmic.gui.qt.niftyigi/bundle/index.html");
}


//-----------------------------------------------------------------------------
void QmitkNiftyIGIApplicationPlugin::start(ctkPluginContext* context)
{
  berry::AbstractUICTKPlugin::start(context);
  this->SetPluginContext(context);

  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyIGIApplication, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewIGIPerspective, context);

  this->RegisterHelpSystem();
}


//-----------------------------------------------------------------------------
void QmitkNiftyIGIApplicationPlugin::stop(ctkPluginContext* context)
{
}


//-----------------------------------------------------------------------------
Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_gui_qt_niftyigi, QmitkNiftyIGIApplicationPlugin)
