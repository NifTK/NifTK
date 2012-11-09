/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-05 06:46:30 +0000 (Sat, 05 Nov 2011) $
 Revision          : $Revision: 7703 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "QmitkNiftyViewApplicationPlugin.h"
#include "QmitkNiftyViewIGIPerspective.h"
#include "QmitkNiftyViewMIDASPerspective.h"
#include "QmitkNiftyViewApplicationPreferencePage.h"
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
  this->context = context;

  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewApplication, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewIGIPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewMIDASPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewApplicationPreferencePage, context);

  this->RegisterMIDASGlobalInteractionPatterns();
  this->RegisterHelpSystem();
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
