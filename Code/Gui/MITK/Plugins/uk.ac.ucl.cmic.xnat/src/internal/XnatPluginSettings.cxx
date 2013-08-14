/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "XnatPluginSettings.h"

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

QString XnatPluginSettings::getDefaultURL() const
{
  std::string server = preferences->Get(XnatPluginPreferencePage::SERVER_NAME, XnatPluginPreferencePage::SERVER_DEFAULT);
  return QString::fromStdString(server);
}

void XnatPluginSettings::setDefaultURL(const QString& url)
{
  preferences->Put(XnatPluginPreferencePage::SERVER_NAME, url.toStdString());
}

QString XnatPluginSettings::getDefaultUserID() const
{
  std::string user = preferences->Get(XnatPluginPreferencePage::USER_NAME, XnatPluginPreferencePage::USER_DEFAULT);
  return QString::fromStdString(user);
}

void XnatPluginSettings::setDefaultUserID(const QString& userID)
{
  preferences->Put(XnatPluginPreferencePage::USER_NAME, userID.toStdString());
}

QString XnatPluginSettings::getDefaultDirectory() const
{
  std::string downloadDirectory = preferences->Get(XnatPluginPreferencePage::DOWNLOAD_DIRECTORY_NAME, XnatPluginPreferencePage::DOWNLOAD_DIRECTORY_DEFAULT);
  return QString::fromStdString(downloadDirectory);
}

void XnatPluginSettings::setDefaultDirectory(const QString& downloadDirectory)
{
  preferences->Put(XnatPluginPreferencePage::DOWNLOAD_DIRECTORY_NAME, downloadDirectory.toStdString());
}

QString XnatPluginSettings::getDefaultWorkDirectory() const
{
  std::string workDirectory = preferences->Get(XnatPluginPreferencePage::WORK_DIRECTORY_NAME, XnatPluginPreferencePage::WORK_DIRECTORY_DEFAULT);
  return QString::fromStdString(workDirectory);
}

void XnatPluginSettings::setDefaultWorkDirectory(const QString& workDirectory)
{
  preferences->Put(XnatPluginPreferencePage::WORK_DIRECTORY_NAME, workDirectory.toStdString());
}

QMap<QString, ctkXnatLoginProfile*> XnatPluginSettings::getLoginProfiles() const
{
  QMap<QString, ctkXnatLoginProfile*> profiles;

  berry::IPreferences::Pointer profilesNode = preferences->Node("profiles");
  std::vector<std::string> profileNames = profilesNode->ChildrenNames();
  std::vector<std::string>::iterator itProfileNames = profileNames.begin();
  std::vector<std::string>::iterator endProfileNames = profileNames.end();
  while (itProfileNames != endProfileNames)
  {
    std::string profileName = *itProfileNames;
    QString qProfileName = QString::fromStdString(profileName);
    berry::IPreferences::Pointer profileNode = profilesNode->Node(profileName);
    ctkXnatLoginProfile* profile = new ctkXnatLoginProfile();
    profile->setName(QString::fromStdString(profileName));
    profile->setServerUri(QString::fromStdString(profileNode->Get("serverUri", "")));
    profile->setUserName(QString::fromStdString(profileNode->Get("userName", "")));
    profile->setPassword(QString::fromStdString(profileNode->Get("password", "")));
    profile->setDefault(profileNode->GetBool("default", false));
    profiles[qProfileName] = profile;
    ++itProfileNames;
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
    berry::IPreferences::Pointer profileNode = profilesNode->Node(profileName.toStdString());
    profileNode->Put("serverUri", profile->serverUri().toStdString());
    profileNode->Put("userName", profile->userName().toStdString());
    // Saving passwords is disabled.
//    profileNode->Put("password", profile->password().toStdString());
    profileNode->PutBool("default", profile->isDefault());
    ++itProfiles;
  }
}

ctkXnatLoginProfile* XnatPluginSettings::getLoginProfile(QString profileName) const
{
  berry::IPreferences::Pointer profilesNode = preferences->Node("profiles");
  berry::IPreferences::Pointer profileNode = profilesNode->Node(profileName.toStdString());
  ctkXnatLoginProfile* profile = new ctkXnatLoginProfile();
  profile->setName(profileName);
  profile->setServerUri(QString::fromStdString(profileNode->Get("serverUri", "")));
  profile->setUserName(QString::fromStdString(profileNode->Get("userName", "")));
  profile->setPassword(QString::fromStdString(profileNode->Get("password", "")));
  profile->setDefault(profileNode->GetBool("default", false));

  return profile;
}

void XnatPluginSettings::setLoginProfile(QString profileName, ctkXnatLoginProfile* profile)
{
  berry::IPreferences::Pointer profilesNode = preferences->Node("profiles");
  berry::IPreferences::Pointer profileNode = profilesNode->Node(profile->name().toStdString());
  profileNode->Put("serverUri", profile->serverUri().toStdString());
  profileNode->Put("userName", profile->userName().toStdString());

  // Saving passwords is disabled.
//  profileNode->Put("password", profile->password().toStdString());
  profileNode->PutBool("default", profile->isDefault());
}

void XnatPluginSettings::removeLoginProfile(QString profileName)
{
  berry::IPreferences::Pointer profilesNode = preferences->Node("profiles");
  berry::IPreferences::Pointer profileNode = profilesNode->Node(profileName.toStdString());
  profileNode->RemoveNode();
}

ctkXnatLoginProfile* XnatPluginSettings::getDefaultLoginProfile() const
{
  QMap<QString, ctkXnatLoginProfile*> profiles = getLoginProfiles();
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
