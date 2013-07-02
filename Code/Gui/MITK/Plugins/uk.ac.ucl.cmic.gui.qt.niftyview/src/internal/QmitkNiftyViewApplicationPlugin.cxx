/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkNiftyViewApplicationPlugin.h"
#include <QmitkNiftyViewIGIPerspective.h>
#include <QmitkNiftyViewMIDASPerspective.h>
#include <QmitkNiftyViewApplicationPreferencePage.h>
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
  return QString("qthelp://uk.ac.ucl.cmic.gui.qt.niftyview/bundle/index.html");
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPlugin::start(ctkPluginContext* context)
{
  berry::AbstractUICTKPlugin::start(context);
  this->SetPluginContext(context);

  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewApplication, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewIGIPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewMIDASPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewApplicationPreferencePage, context);

  this->RegisterHelpSystem();
  this->RegisterMIDASGlobalInteractionPatterns();
  this->RegisterDataStorageListener();
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPlugin::stop(ctkPluginContext* context)
{
  this->UnregisterDataStorageListener();
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPlugin::NodeAdded(const mitk::DataNode *constNode)
{
  mitk::DataNode::Pointer node = const_cast<mitk::DataNode*>(constNode);
  this->RegisterLevelWindowProperty("uk.ac.ucl.cmic.gui.qt.niftyview", node);
}


//-----------------------------------------------------------------------------
Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_gui_qt_niftyview, QmitkNiftyViewApplicationPlugin)
