#include "XnatPluginSettings.h"

#include <QDir>
#include <QFileInfo>
#include <QUuid>

#include "XnatPluginPreferencePage.h"

XnatPluginSettings::XnatPluginSettings(berry::IPreferences::Pointer preferences)
: XnatBrowserSettings()
, preferences(preferences)
{
}

QString XnatPluginSettings::getDefaultURL()
{
  // initialize output XNAT URL
  QString url;

  // TODO

  return url;
}

void XnatPluginSettings::setDefaultURL(const QString& url)
{
  // TODO
}

QString XnatPluginSettings::getDefaultUserID()
{
  // initialize output XNAT user identifier
  QString userID;

  // TODO

  return userID;
}

void XnatPluginSettings::setDefaultUserID(const QString& userID)
{
  // TODO
}

QString XnatPluginSettings::getDefaultDirectory()
{
  std::string downloadDirectory = preferences->Get(XnatPluginPreferencePage::DOWNLOAD_DIRECTORY_NAME, XnatPluginPreferencePage::DOWNLOAD_DIRECTORY_DEFAULT);

  return QString::fromStdString(downloadDirectory);
}

void XnatPluginSettings::setDefaultDirectory(const QString& downloadDirectory)
{
  preferences->Put(XnatPluginPreferencePage::DOWNLOAD_DIRECTORY_NAME, downloadDirectory.toStdString());
}

QString XnatPluginSettings::getDefaultWorkDirectory()
{
  // initialize output work directory name
  QString workDir;

  // TODO

  return workDir;
}

void XnatPluginSettings::setDefaultWorkDirectory(const QString& workDir)
{
  // TODO
}

QString XnatPluginSettings::getWorkSubdirectory()
{
  // set work directory name
  QDir workDir;
  QString workDirName = getDefaultWorkDirectory();
  if ( !workDirName.isEmpty() )
  {
    workDir = QDir(workDirName);
  }

  // generate random name for subdirectory
  QString subdir = QUuid::createUuid().toString();

  // create subdirectory in work directory
  bool subdirCreated = workDir.mkdir(subdir);

  // check whether subdirectory was created
  if ( !subdirCreated )
  {
    // display error message
    return QString();
  }

  // return full path of subdirectory
  return QFileInfo(workDir, subdir).absoluteFilePath();
}

