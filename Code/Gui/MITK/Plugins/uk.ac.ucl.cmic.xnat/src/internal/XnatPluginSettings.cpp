#include "XnatPluginSettings.h"

#include <mitkLogMacros.h>

#include <QDir>
#include <QFileInfo>
#include <QUuid>

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
  MITK_INFO << "XnatPluginSettings::getLoginProfiles() const";
  QMap<QString, XnatLoginProfile*> result;
  return result;
}

void XnatPluginSettings::setLoginProfiles(QMap<QString, XnatLoginProfile*> loginProfiles)
{
  MITK_INFO << "setLoginProfiles(QMap<QString, XnatLoginProfile*> loginProfiles)";
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
