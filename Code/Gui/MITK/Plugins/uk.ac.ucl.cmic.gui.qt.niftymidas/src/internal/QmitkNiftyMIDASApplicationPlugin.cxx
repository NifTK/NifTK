/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkNiftyMIDASApplicationPlugin.h"
#include <QmitkNiftyViewIGIPerspective.h>
#include <QmitkNiftyViewMIDASPerspective.h>
#include <QmitkNiftyViewApplicationPreferencePage.h>
#include "../QmitkNiftyMIDASApplication.h"

//-----------------------------------------------------------------------------
QmitkNiftyMIDASApplicationPlugin::QmitkNiftyMIDASApplicationPlugin()
{
}


//-----------------------------------------------------------------------------
QmitkNiftyMIDASApplicationPlugin::~QmitkNiftyMIDASApplicationPlugin()
{
}


//-----------------------------------------------------------------------------
QString QmitkNiftyMIDASApplicationPlugin::GetHelpHomePageURL() const
{
  return QString("qthelp://uk.ac.ucl.cmic.gui.qt.niftymidas/bundle/index.html");
}


//-----------------------------------------------------------------------------
void QmitkNiftyMIDASApplicationPlugin::start(ctkPluginContext* context)
{
  berry::AbstractUICTKPlugin::start(context);
  this->SetPluginContext(context);

  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyMIDASApplication, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewIGIPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewMIDASPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewApplicationPreferencePage, context);

  this->RegisterHelpSystem();
  this->RegisterMIDASGlobalInteractionPatterns();
  this->RegisterDataStorageListener();
  this->SetFileOpenTriggersReinit(false);
}


//-----------------------------------------------------------------------------
void QmitkNiftyMIDASApplicationPlugin::stop(ctkPluginContext* context)
{
  this->UnregisterDataStorageListener();
}


//-----------------------------------------------------------------------------
void QmitkNiftyMIDASApplicationPlugin::NodeAdded(const mitk::DataNode *constNode)
{
  mitk::DataNode::Pointer node = const_cast<mitk::DataNode*>(constNode);
  this->RegisterLevelWindowProperty("uk.ac.ucl.cmic.gui.qt.niftymidas", node);
}


//-----------------------------------------------------------------------------
Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_gui_qt_niftymidas, QmitkNiftyMIDASApplicationPlugin)
