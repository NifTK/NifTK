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
#include "QmitkNiftyViewCMICPerspective.h"
#include "QmitkNiftyViewMIDASPerspective.h"
#include "../QmitkNiftyViewApplication.h"

#include <mitkVersion.h>
#include <berryQtAssistantUtil.h>

#include <QFileInfo>
#include <QDateTime>
#include <QtPlugin>


#include "mitkGlobalInteraction.h"
#include "NifTKConfigure.h"

const std::string QmitkNiftyViewApplicationPlugin::MIDAS_KEYPRESS_STATE_MACHINE_XML = std::string(
    "      <stateMachine NAME=\"MIDASKeyPressStateMachine\">"
    "         <state NAME=\"stateStart\"  START_STATE=\"TRUE\"   ID=\"1\" X_POS=\"50\"   Y_POS=\"100\" WIDTH=\"100\" HEIGHT=\"50\">"
    "           <transition NAME=\"keyPressA\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4001\">"
    "             <action ID=\"350001\" />"
    "           </transition>"
    "           <transition NAME=\"keyPressZ\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4019\">"
    "             <action ID=\"350002\" />"
    "           </transition>"
    "           <transition NAME=\"keyPressQ\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4013\">"
    "             <action ID=\"350003\" />"
    "           </transition>"
    "           <transition NAME=\"keyPressW\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4016\">"
    "             <action ID=\"350004\" />"
    "           </transition>"
    "           <transition NAME=\"keyPressE\" NEXT_STATE_ID=\"1\" EVENT_ID=\"19\">"
    "             <action ID=\"350005\" />"
    "           </transition>"
    "         </state>"
    "      </stateMachine>"
  );

const std::string QmitkNiftyViewApplicationPlugin::MIDAS_SEED_DROPPER_STATE_MACHINE_XML = std::string(
"      <stateMachine NAME=\"MIDASSeedDropper\">"
"         <state NAME=\"stateStart\"  START_STATE=\"TRUE\"   ID=\"1\" X_POS=\"50\"   Y_POS=\"100\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"rightButtonDown\" NEXT_STATE_ID=\"1\" EVENT_ID=\"2\">"
"             <!-- 10 = AcADDPOINT  -->"
"             <action ID=\"10\" />"
"             <!-- 72 = AcDESELECTALL  -->"
"             <action ID=\"72\" />"
"           </transition>"
"         </state>"
"      </stateMachine>"
);

const std::string QmitkNiftyViewApplicationPlugin::MIDAS_SEED_TOOL_STATE_MACHINE_XML = std::string(
"      <stateMachine NAME=\"MIDASSeedTool\">"
"         <state NAME=\"stateStart\"  START_STATE=\"TRUE\"   ID=\"1\" X_POS=\"50\"   Y_POS=\"100\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"middleButtonDown\" NEXT_STATE_ID=\"2\" EVENT_ID=\"4\">"
"             <!-- 30 = AcCHECKELEMENT  -->"
"             <action ID=\"30\" />"
"           </transition>"
"           <transition NAME=\"middleButtonDownMouseMove\" NEXT_STATE_ID=\"2\" EVENT_ID=\"533\">"
"             <!-- 30 = AcCHECKELEMENT  -->"
"             <action ID=\"30\" />"
"           </transition>"
"           <transition NAME=\"leftButtonDown\" NEXT_STATE_ID=\"3\" EVENT_ID=\"1\">"
"             <!-- 30 = AcCHECKELEMENT  -->"
"             <action ID=\"30\" />"
"           </transition>"
"           <transition NAME=\"leftButtonDownMouseMove\" NEXT_STATE_ID=\"1\" EVENT_ID=\"530\">"
"             <!-- 91 = AcMOVESELECTED  -->"
"             <action ID=\"91\" />"
"           </transition>"
"           <transition NAME=\"leftButtonUp\" NEXT_STATE_ID=\"1\" EVENT_ID=\"505\">"
"             <!-- 42 = AcFINISHMOVEMENT  -->"
"             <action ID=\"42\" />"
"             <!-- 72 = AcDESELECTALL  -->"
"             <action ID=\"72\" />"
"           </transition>"
"         </state>"
"         <state NAME=\"guardMiddleButtonPointSelected\"   ID=\"2\" X_POS=\"100\" Y_POS=\"150\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"no\" NEXT_STATE_ID=\"1\" EVENT_ID=\"1003\">"
"             <!-- 0 = AcDONOTHING -->"
"             <action ID=\"0\" />"
"           </transition>"
"           <transition NAME=\"yes\" NEXT_STATE_ID=\"1\" EVENT_ID=\"1004\">"
"             <!-- 100 = AcREMOVEPOINT -->"
"             <action ID=\"100\" />"
"             <!-- 72 = AcDESELECTALL  -->"
"             <action ID=\"72\" />"
"           </transition>"
"         </state>"
"         <state NAME=\"guardLeftButtonPointSelected\"     ID=\"3\" X_POS=\"100\" Y_POS=\"50\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"no\" NEXT_STATE_ID=\"1\" EVENT_ID=\"1003\">"
"             <!-- 0 = AcDONOTHING -->"
"             <action ID=\"0\" />"
"           </transition>"
"           <transition NAME=\"yes\" NEXT_STATE_ID=\"1\" EVENT_ID=\"1004\">"
"             <!-- 8 = AcINITMOVEMENT -->"
"             <action ID=\"8\" />"
"             <!-- 60 = AcSELECTPICKEDOBJECT -->"
"             <action ID=\"60\" />"
"           </transition>"
"         </state>"
"      </stateMachine>"
);

const std::string QmitkNiftyViewApplicationPlugin::MIDAS_POLY_TOOL_STATE_MACHINE_XML = std::string(
"      <stateMachine NAME=\"MIDASPolyTool\">"
"         <state NAME=\"stateStart\"  START_STATE=\"TRUE\"   ID=\"1\" X_POS=\"50\"   Y_POS=\"100\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"leftButtonDown\" NEXT_STATE_ID=\"1\" EVENT_ID=\"1\">"
"             <!-- 12 = AcADDLINE  -->"
"             <action ID=\"12\" />"
"           </transition>"
"           <transition NAME=\"middleButtonDown\" NEXT_STATE_ID=\"2\" EVENT_ID=\"4\">"
"             <!-- 66 = AcSELECTPOINT  -->"
"             <action ID=\"66\" />"
"           </transition>"
"         </state>"
"         <state NAME=\"movingLine\"   ID=\"2\" X_POS=\"100\" Y_POS=\"100\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"middleButtonDownMouseMove\" NEXT_STATE_ID=\"2\" EVENT_ID=\"533\">"
"             <!-- 90 = AcMOVEPOINT  -->"
"             <action ID=\"90\" />"
"           </transition>"
"           <transition NAME=\"middleButtonUp\" NEXT_STATE_ID=\"1\" EVENT_ID=\"506\">"
"             <!-- 76 = AcDESELECTPOINT  -->"
"             <action ID=\"76\" />"
"           </transition>"
"         </state>"
"      </stateMachine>"
);

const std::string QmitkNiftyViewApplicationPlugin::MIDAS_DRAW_TOOL_STATE_MACHINE_XML = std::string(
"      <stateMachine NAME=\"MIDASDrawTool\">"
"         <state NAME=\"stateStart\"  START_STATE=\"TRUE\"   ID=\"1\" X_POS=\"50\"   Y_POS=\"100\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"leftButtonDown\" NEXT_STATE_ID=\"1\" EVENT_ID=\"1\">"
"             <!-- 80 = AcNEWPOINT  -->"
"             <action ID=\"80\" />"
"           </transition>"
"           <transition NAME=\"leftButtonUp\" NEXT_STATE_ID=\"1\" EVENT_ID=\"505\">"
"             <!-- 42 = AcFINISHMOVEMENT  -->"
"             <action ID=\"42\" />"
"           </transition>"
"           <transition NAME=\"leftButtonDownMouseMove\" NEXT_STATE_ID=\"1\" EVENT_ID=\"530\">"
"             <!-- 90 = AcMOVEPOINT  -->"
"             <action ID=\"90\" />"
"           </transition>"
"         </state>"
"      </stateMachine>"
);


// Note: In MIDAS, left button, adds to segmentation image.
// Note: In MIDAS, middle button, adds to mask that influences connection breaker.
// Note: In MIDAS, right button, subtracts from the mask that influences connection breaker.
// So, we just add shift to distinguish from normal MITK interaction.

const std::string QmitkNiftyViewApplicationPlugin::MIDAS_PAINTBRUSH_TOOL_STATE_MACHINE_XML = std::string(
"      <stateMachine NAME=\"MIDASPaintbrushTool\">"
"         <state NAME=\"stateStart\"  START_STATE=\"TRUE\"   ID=\"1\" X_POS=\"50\"   Y_POS=\"100\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"leftButtonDown\" NEXT_STATE_ID=\"1\" EVENT_ID=\"1\">"
"             <action ID=\"320401\" />"
"           </transition>"
"           <transition NAME=\"leftButtonUp\" NEXT_STATE_ID=\"1\" EVENT_ID=\"505\">"
"             <action ID=\"320402\" />"
"           </transition>"
"           <transition NAME=\"leftButtonDownMouseMove\" NEXT_STATE_ID=\"1\" EVENT_ID=\"530\">"
"             <action ID=\"320403\" />"
"           </transition>"
"           <transition NAME=\"middleButtonDown\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4\">"
"             <action ID=\"320404\" />"
"           </transition>"
"           <transition NAME=\"middleButtonUp\" NEXT_STATE_ID=\"1\" EVENT_ID=\"506\">"
"             <action ID=\"320405\" />"
"           </transition>"
"           <transition NAME=\"middleButtonDownMouseMove\" NEXT_STATE_ID=\"1\" EVENT_ID=\"533\">"
"             <action ID=\"320406\" />"
"           </transition>"
"           <transition NAME=\"rightButtonDown\" NEXT_STATE_ID=\"1\" EVENT_ID=\"2\">"
"             <action ID=\"320407\" />"
"           </transition>"
"           <transition NAME=\"rightButtonUp\" NEXT_STATE_ID=\"1\" EVENT_ID=\"507\">"
"             <action ID=\"320408\" />"
"           </transition>"
"           <transition NAME=\"rightButtonDownMouseMove\" NEXT_STATE_ID=\"1\" EVENT_ID=\"531\">"
"             <action ID=\"320409\" />"
"           </transition>"
"         </state>"
"      </stateMachine>"
);

QmitkNiftyViewApplicationPlugin* QmitkNiftyViewApplicationPlugin::inst = 0;

QmitkNiftyViewApplicationPlugin::QmitkNiftyViewApplicationPlugin()
  : pluginListener(0)
{
  inst = this;
}

QmitkNiftyViewApplicationPlugin::~QmitkNiftyViewApplicationPlugin()
{
  delete pluginListener;
}

QmitkNiftyViewApplicationPlugin* QmitkNiftyViewApplicationPlugin::GetDefault()
{
  return inst;
}

void QmitkNiftyViewApplicationPlugin::start(ctkPluginContext* context)
{
  berry::AbstractUICTKPlugin::start(context);
  
  this->context = context;
  
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewCMICPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewMIDASPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewApplication, context);

  QString collectionFile = GetQtHelpCollectionFile();

  berry::QtAssistantUtil::SetHelpCollectionFile(collectionFile);
  berry::QtAssistantUtil::SetDefaultHelpUrl("qthelp://uk.ac.ucl.cmic.gui.qt.niftyview/bundle/index.html");

  delete pluginListener;
  pluginListener = new berry::QCHPluginListener(context);
  context->connectPluginListener(pluginListener, SLOT(pluginChanged(ctkPluginEvent)), Qt::DirectConnection);

  // register all QCH files from all the currently installed plugins
  pluginListener->processPlugins();


  // Load StateMachine patterns
  mitk::GlobalInteraction* globalInteractor =  mitk::GlobalInteraction::GetInstance();
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(MIDAS_SEED_DROPPER_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(MIDAS_SEED_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(MIDAS_DRAW_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(MIDAS_POLY_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(MIDAS_PAINTBRUSH_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(MIDAS_KEYPRESS_STATE_MACHINE_XML);

  berry::IPreferencesService::Pointer prefService =
  berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  Poco::Path logoPath(berry::Platform::GetInstancePath(), "Blank.png");
  berry::IPreferences::Pointer logoPref = prefService->GetSystemPreferences()->Node("DepartmentLogo");
  logoPref->Put("DepartmentLogo", logoPath.toString().c_str());
}

ctkPluginContext* QmitkNiftyViewApplicationPlugin::GetPluginContext() const
{
  return context;
}

QString QmitkNiftyViewApplicationPlugin::GetQtHelpCollectionFile() const
{
  if (!helpCollectionFile.isEmpty())
  {
    return helpCollectionFile;
  }

  QString collectionFilename;
  QString na("n/a");
  if (na != MITK_REVISION)
  {
    collectionFilename = "CMICNiftyViewQtHelpCollection_" MITK_REVISION ".qhc";
  }
  else
  {
    collectionFilename = "CMICNiftyViewQtHelpCollection.qhc";
  }

  QFileInfo collectionFileInfo = context->getDataFile(collectionFilename);
  QFileInfo pluginFileInfo = QFileInfo(QUrl(context->getPlugin()->getLocation()).toLocalFile());
  if (!collectionFileInfo.exists() ||
      pluginFileInfo.lastModified() > collectionFileInfo.lastModified())
  {
    // extract the qhc file from the plug-in
    QByteArray content = context->getPlugin()->getResource(collectionFilename);
    if (content.isEmpty())
    {
      BERRY_WARN << "Could not get plug-in resource: " << collectionFilename.toStdString();
    }
    else
    {
      QFile file(collectionFileInfo.absoluteFilePath());
      file.open(QIODevice::WriteOnly);
      file.write(content);
      file.close();
    }
  }

  if (QFile::exists(collectionFileInfo.absoluteFilePath()))
  {
    helpCollectionFile = collectionFileInfo.absoluteFilePath();
  }

  return helpCollectionFile;
}

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_gui_qt_niftyview, QmitkNiftyViewApplicationPlugin)
