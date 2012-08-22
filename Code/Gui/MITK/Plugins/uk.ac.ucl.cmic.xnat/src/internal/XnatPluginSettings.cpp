#include "XnatPluginSettings.h"

#include <mitkLogMacros.h>

#include <QDir>
#include <QFileInfo>
#include <QMap>
#include <QUuid>

#include <vector>

#include <XnatLoginProfile.h>

#include "XnatPluginPreferencePage.h"

XnatPluginSettings::XnatPluginSettings(berry::IPreferences::Pointer preferences)
: XnatSettings()
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

QMap<QString, XnatLoginProfile*> XnatPluginSettings::getLoginProfiles() const
{
  QMap<QString, XnatLoginProfile*> profiles;

  berry::IPreferences::Pointer profilesNode = preferences->Node("profiles");
  std::vector<std::string> profileNames = profilesNode->ChildrenNames();
  std::vector<std::string>::iterator itProfileNames = profileNames.begin();
  std::vector<std::string>::iterator endProfileNames = profileNames.end();
  while (itProfileNames != endProfileNames)
  {
    std::string profileName = *itProfileNames;
    QString qProfileName = QString::fromStdString(profileName);
    berry::IPreferences::Pointer profileNode = profilesNode->Node(profileName);
    XnatLoginProfile* profile = new XnatLoginProfile();
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

void XnatPluginSettings::setLoginProfiles(QMap<QString, XnatLoginProfile*> loginProfiles)
{
  berry::IPreferences::Pointer profilesNode = preferences->Node("profiles");
  QMap<QString, XnatLoginProfile*>::iterator itProfiles = loginProfiles.begin();
  QMap<QString, XnatLoginProfile*>::iterator endProfiles = loginProfiles.end();
  while (itProfiles != endProfiles)
  {
    QString profileName = itProfiles.key();
    XnatLoginProfile* profile = itProfiles.value();
    berry::IPreferences::Pointer profileNode = profilesNode->Node(profileName.toStdString());
    profileNode->Put("serverUri", profile->serverUri().toStdString());
    profileNode->Put("userName", profile->userName().toStdString());
    profileNode->Put("password", profile->password().toStdString());
    profileNode->PutBool("default", profile->isDefault());
    ++itProfiles;
  }
}

XnatLoginProfile* XnatPluginSettings::getLoginProfile(QString profileName) const
{
  MITK_INFO << "XnatPluginSettings::getLoginProfile(QString profileName) const";
  return 0;
}

void XnatPluginSettings::setLoginProfile(QString profileName, XnatLoginProfile*)
{
  MITK_INFO << "XnatPluginSettings::setLoginProfile(QString profileName, XnatLoginProfile*)";
}

XnatLoginProfile* XnatPluginSettings::getDefaultLoginProfile() const
{
  QMap<QString, XnatLoginProfile*> profiles = getLoginProfiles();
  QMap<QString, XnatLoginProfile*>::const_iterator itProfiles = profiles.begin();
  QMap<QString, XnatLoginProfile*>::const_iterator endProfiles = profiles.end();
  while (itProfiles != endProfiles)
  {
    XnatLoginProfile* profile = itProfiles.value();
    if (profile->isDefault())
    {
      return profile;
    }
    ++itProfiles;
  }
  return 0;
}
