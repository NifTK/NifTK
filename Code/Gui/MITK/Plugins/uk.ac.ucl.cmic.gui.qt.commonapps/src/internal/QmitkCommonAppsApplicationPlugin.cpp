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

#include "QmitkCommonAppsApplicationPlugin.h"
#include "QmitkCommonAppsApplicationPreferencePage.h"

#include <mitkProperties.h>
#include <mitkVersion.h>
#include <mitkLogMacros.h>
#include <mitkDataStorage.h>
#include <mitkVtkResliceInterpolationProperty.h>

#include <service/cm/ctkConfigurationAdmin.h>
#include <service/cm/ctkConfiguration.h>

#include <QFileInfo>
#include <QDateTime>
#include <QtPlugin>

#include "NifTKConfigure.h"

QmitkCommonAppsApplicationPlugin* QmitkCommonAppsApplicationPlugin::inst = 0;

//-----------------------------------------------------------------------------
QmitkCommonAppsApplicationPlugin::QmitkCommonAppsApplicationPlugin()
: context(NULL)
, m_DataStorageServiceTracker(NULL)
, m_InDataStorageChanged(false)
{
  inst = this;
}


//-----------------------------------------------------------------------------
QmitkCommonAppsApplicationPlugin::~QmitkCommonAppsApplicationPlugin()
{
}


//-----------------------------------------------------------------------------
QmitkCommonAppsApplicationPlugin* QmitkCommonAppsApplicationPlugin::GetDefault()
{
  return inst;
}


//-----------------------------------------------------------------------------
const mitk::DataStorage* QmitkCommonAppsApplicationPlugin::GetDataStorage()
{
  mitk::DataStorage::Pointer dataStorage = NULL;

  if (m_DataStorageServiceTracker != NULL)
  {
    mitk::IDataStorageService* dsService = m_DataStorageServiceTracker->getService();
    if (dsService != 0)
    {
      dataStorage = dsService->GetDataStorage()->GetDataStorage();
    }
  }

  return dataStorage;
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::start(ctkPluginContext* context)
{
  berry::AbstractUICTKPlugin::start(context);

  BERRY_REGISTER_EXTENSION_CLASS(QmitkCommonAppsApplicationPreferencePage, context);

  this->context = context;
  this->m_DataStorageServiceTracker = new ctkServiceTracker<mitk::IDataStorageService*>(context);
  this->m_DataStorageServiceTracker->open();
  
  this->GetDataStorage()->AddNodeEvent.AddListener
      ( mitk::MessageDelegate1<QmitkCommonAppsApplicationPlugin, const mitk::DataNode*>
        ( this, &QmitkCommonAppsApplicationPlugin::NodeAddedProxy ) );

  ctkServiceReference cmRef = context->getServiceReference<ctkConfigurationAdmin>();
  ctkConfigurationAdmin* configAdmin = 0;
  if (cmRef)
  {
    configAdmin = context->getService<ctkConfigurationAdmin>(cmRef);
  }

  // Use the CTK Configuration Admin service to configure the BlueBerry help system
  if (configAdmin)
  {
    ctkConfigurationPtr conf = configAdmin->getConfiguration("org.blueberry.services.help", QString());
    ctkDictionary helpProps;
    QString urlHomePage = this->GetHelpHomePageURL();
    helpProps.insert("homePage", urlHomePage);
    conf->update(helpProps);
    context->ungetService(cmRef);
  }
  else
  {
    MITK_WARN << "Configuration Admin service unavailable, cannot set home page url.";
  }

  // Blank the departmental logo for now.
  berry::IPreferencesService::Pointer prefService =
  berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  berry::IPreferences::Pointer logoPref = prefService->GetSystemPreferences()->Node("org.mitk.editors.stdmultiwidget");
  logoPref->Put("DepartmentLogo", "");
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::stop(ctkPluginContext* context)
{

  if (m_DataStorageServiceTracker != NULL)
  {

    this->GetDataStorage()->AddNodeEvent.RemoveListener
        ( mitk::MessageDelegate1<QmitkCommonAppsApplicationPlugin, const mitk::DataNode*>
          ( this, &QmitkCommonAppsApplicationPlugin::NodeAddedProxy ) );

    m_DataStorageServiceTracker->close();
    delete m_DataStorageServiceTracker;
    m_DataStorageServiceTracker = NULL;
  }
}


//-----------------------------------------------------------------------------
ctkPluginContext* QmitkCommonAppsApplicationPlugin::GetPluginContext() const
{
  return context;
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::NodeAddedProxy(const mitk::DataNode *node)
{
  // guarantee no recursions when a new node event is thrown in NodeAdded()
  if(!m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    this->NodeAdded(node);
    m_InDataStorageChanged = false;
  }
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::NodeAdded(const mitk::DataNode *constNode)
{
  mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(constNode->GetData());
  if (image.IsNotNull())
  {
    bool isBinary(true);
    constNode->GetBoolProperty("binary", isBinary);

    if (isBinary)
    {
      return;
    }

    berry::IPreferencesService::Pointer prefService =
    berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

    berry::IPreferences::Pointer prefNode = prefService->GetSystemPreferences()->Node("uk.ac.ucl.cmic.gui.qt.commonapps");
    int imageResliceInterpolation =  prefNode->GetInt(QmitkCommonAppsApplicationPreferencePage::IMAGE_RESLICE_INTERPOLATION, 2);
    int imageTextureInterpolation =  prefNode->GetInt(QmitkCommonAppsApplicationPreferencePage::IMAGE_TEXTURE_INTERPOLATION, 2);

    mitk::DataNode::Pointer node = const_cast<mitk::DataNode*>(constNode);

    if (imageTextureInterpolation == 0)
    {
      node->SetProperty("texture interpolation", mitk::BoolProperty::New(false));
    }
    else
    {
      node->SetProperty("texture interpolation", mitk::BoolProperty::New(true));
    }

    mitk::VtkResliceInterpolationProperty::Pointer interpolationProperty = mitk::VtkResliceInterpolationProperty::New();

    if (imageResliceInterpolation == 0)
    {
      interpolationProperty->SetInterpolationToNearest();
    }
    else if (imageResliceInterpolation == 1)
    {
      interpolationProperty->SetInterpolationToLinear();
    }
    else if (imageResliceInterpolation == 2)
    {
      interpolationProperty->SetInterpolationToCubic();
    }
    node->SetProperty("reslice interpolation", interpolationProperty);

    node->SetProperty("black opacity", mitk::FloatProperty::New(1));
  }
}

//-----------------------------------------------------------------------------
Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_gui_qt_commonapps, QmitkCommonAppsApplicationPlugin)
