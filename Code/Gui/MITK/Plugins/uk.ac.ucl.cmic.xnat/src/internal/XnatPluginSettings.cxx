/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "XnatPluginSettings.h"

#include <berryIPreferences.h>

#include <mitkLogMacros.h>

#include <QDir>
#include <QFileInfo>
#include <QMap>
#include <QUuid>

#include <vector>

#include <ctkXnatLoginProfile.h>

#include "XnatPluginPreferencePage.h"

XnatPluginSettings::XnatPluginSettings(berry::IPreferences::Pointer preferences)
: ctkXnatSettings()
, preferences(preferences)
{
}

QString XnatPluginSettings::defaultURL() const
{
  return preferences->Get(XnatPluginPreferencePage::SERVER_NAME, XnatPluginPreferencePage::SERVER_DEFAULT);
}

void XnatPluginSettings::setDefaultURL(const QString& url)
{
  preferences->Put(XnatPluginPreferencePage::SERVER_NAME, url);
}

QString XnatPluginSettings::defaultUserID() const
{
  return preferences->Get(XnatPluginPreferencePage::USER_NAME, XnatPluginPreferencePage::USER_DEFAULT);
}

void XnatPluginSettings::setDefaultUserID(const QString& userID)
{
  preferences->Put(XnatPluginPreferencePage::USER_NAME, userID);
}

QString XnatPluginSettings::defaultDirectory() const
{
  return preferences->Get(XnatPluginPreferencePage::DOWNLOAD_DIRECTORY_NAME, XnatPluginPreferencePage::DOWNLOAD_DIRECTORY_DEFAULT);
}

void XnatPluginSettings::setDefaultDirectory(const QString& downloadDirectory)
{
  preferences->Put(XnatPluginPreferencePage::DOWNLOAD_DIRECTORY_NAME, downloadDirectory);
}

QString XnatPluginSettings::defaultWorkDirectory() const
{
  return preferences->Get(XnatPluginPreferencePage::WORK_DIRECTORY_NAME, XnatPluginPreferencePage::WORK_DIRECTORY_DEFAULT);
}

void XnatPluginSettings::setDefaultWorkDirectory(const QString& workDirectory)
{
  preferences->Put(XnatPluginPreferencePage::WORK_DIRECTORY_NAME, workDirectory);
}

QMap<QString, ctkXnatLoginProfile*> XnatPluginSettings::loginProfiles() const
{
  QMap<QString, ctkXnatLoginProfile*> profiles;

  berry::IPreferences::Pointer profilesNode = preferences->Node("profiles");
  foreach (QString profileName, profilesNode->ChildrenNames())
  {
    berry::IPreferences::Pointer profileNode = profilesNode->Node(profileName);
    ctkXnatLoginProfile* profile = new ctkXnatLoginProfile();
    profile->setName(profileName);
    profile->setServerUrl(profileNode->Get("serverUrl", ""));
    profile->setUserName(profileNode->Get("userName", ""));
    profile->setPassword(profileNode->Get("password", ""));
    profile->setDefault(profileNode->GetBool("default", false));
    profiles[profileName] = profile;
  }

  return profiles;
}

void XnatPluginSettings::setLoginProfiles(QMap<QString, ctkXnatLoginProfile*> loginProfiles)
{
  berry::IPreferences::Pointer profilesNode = preferences->Node("profiles");
  QMap<QString, ctkXnatLoginProfile*>::iterator itProfiles = loginProfiles.begin();
  QMap<QString, ctkXnatLoginProfile*>::iterator endProfiles = loginProfiles.end();
  while (itProfiles != endProfiles)
  {
    QString profileName = itProfiles.key();
    ctkXnatLoginProfile* profile = itProfiles.value();
    berry::IPreferences::Pointer profileNode = profilesNode->Node(profileName);
    profileNode->Put("serverUrl", profile->serverUrl().toString());
    profileNode->Put("userName", profile->userName());
    // Saving passwords is disabled.
//    profileNode->Put("password", profile->password().toStdString());
    profileNode->PutBool("default", profile->isDefault());
    ++itProfiles;
  }
}

ctkXnatLoginProfile* XnatPluginSettings::loginProfile(QString profileName) const
{
  berry::IPreferences::Pointer profilesNode = preferences->Node("profiles");
  berry::IPreferences::Pointer profileNode = profilesNode->Node(profileName);
  ctkXnatLoginProfile* profile = new ctkXnatLoginProfile();
  profile->setName(profileName);
  profile->setServerUrl(profileNode->Get("serverUrl", ""));
  profile->setUserName(profileNode->Get("userName", ""));
  profile->setPassword(profileNode->Get("password", ""));
  profile->setDefault(profileNode->GetBool("default", false));

  return profile;
}

void XnatPluginSettings::setLoginProfile(QString profileName, ctkXnatLoginProfile* profile)
{
  berry::IPreferences::Pointer profilesNode = preferences->Node("profiles");
  berry::IPreferences::Pointer profileNode = profilesNode->Node(profile->name());
  profileNode->Put("serverUrl", profile->serverUrl().toString());
  profileNode->Put("userName", profile->userName());

  // Saving passwords is disabled.
//  profileNode->Put("password", profile->password());
  profileNode->PutBool("default", profile->isDefault());
}

void XnatPluginSettings::removeLoginProfile(QString profileName)
{
  berry::IPreferences::Pointer profilesNode = preferences->Node("profiles");
  berry::IPreferences::Pointer profileNode = profilesNode->Node(profileName);
  profileNode->RemoveNode();
}

ctkXnatLoginProfile* XnatPluginSettings::defaultLoginProfile() const
{
  QMap<QString, ctkXnatLoginProfile*> profiles = this->loginProfiles();
  QMap<QString, ctkXnatLoginProfile*>::const_iterator itProfiles = profiles.begin();
  QMap<QString, ctkXnatLoginProfile*>::const_iterator endProfiles = profiles.end();
  while (itProfiles != endProfiles)
  {
    ctkXnatLoginProfile* profile = itProfiles.value();
    if (profile->isDefault())
    {
      return profile;
    }
    ++itProfiles;
  }
  return 0;
}
