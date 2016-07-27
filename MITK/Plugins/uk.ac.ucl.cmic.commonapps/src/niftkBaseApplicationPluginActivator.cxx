/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseApplicationPluginActivator.h"

#include <service/cm/ctkConfigurationAdmin.h>
#include <service/cm/ctkConfiguration.h>

#include <QString>

#include <berryIPreferences.h>
#include <berryIPreferencesService.h>
#include <berryPlatform.h>

#include <mitkLogMacros.h>


namespace niftk
{

BaseApplicationPluginActivator* BaseApplicationPluginActivator::s_Instance = nullptr;

//-----------------------------------------------------------------------------
BaseApplicationPluginActivator::BaseApplicationPluginActivator()
{
  s_Instance = this;
}


//-----------------------------------------------------------------------------
BaseApplicationPluginActivator::~BaseApplicationPluginActivator()
{
}


//-----------------------------------------------------------------------------
BaseApplicationPluginActivator* BaseApplicationPluginActivator::GetInstance()
{
  return s_Instance;
}


//-----------------------------------------------------------------------------
ctkPluginContext* BaseApplicationPluginActivator::GetContext() const
{
  return m_Context;
}


//-----------------------------------------------------------------------------
void BaseApplicationPluginActivator::start(ctkPluginContext* context)
{
  m_Context = context;
}


//-----------------------------------------------------------------------------
void BaseApplicationPluginActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}


//-----------------------------------------------------------------------------
void BaseApplicationPluginActivator::RegisterHelpSystem()
{
  ctkServiceReference cmRef = m_Context->getServiceReference<ctkConfigurationAdmin>();
  ctkConfigurationAdmin* configAdmin = 0;
  if (cmRef)
  {
    configAdmin = m_Context->getService<ctkConfigurationAdmin>(cmRef);
  }

  // Use the CTK Configuration Admin service to configure the BlueBerry help system
  if (configAdmin)
  {
    ctkConfigurationPtr conf = configAdmin->getConfiguration("org.blueberry.services.help", QString());
    ctkDictionary helpProps;
    QString urlHomePage = this->GetHelpHomePageURL();
    helpProps.insert("homePage", urlHomePage);
    conf->update(helpProps);
    m_Context->ungetService(cmRef);
  }
  else
  {
    MITK_WARN << "Configuration Admin service unavailable, cannot set home page url.";
  }
}


//-----------------------------------------------------------------------------
void BaseApplicationPluginActivator::SetFileOpenTriggersReinit(bool openEditor)
{
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  berry::IPreferences::Pointer generalPrefs = prefService->GetSystemPreferences()->Node("/General");
  generalPrefs->PutBool("OpenEditor", openEditor);
}

}
