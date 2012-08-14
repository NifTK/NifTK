#include "XnatBrowserSettings.h"

#include <QDir>
#include <QFileInfo>
#include <QUuid>

XnatBrowserSettings XnatBrowserSettings::_instance_;

XnatBrowserSettings::XnatBrowserSettings()
: defaultXnatURL( "https://xnat.cbis.jhmi.edu" )
, defaultXnatUserID("")
, defaultDirectory(QDir::currentPath())
, defaultWorkDirectory(QDir::currentPath())
, xnatBrowserGroup("XnatBrowser")
, xnatUrlKey("URL")
, xnatUserIdKey("UserID")
, directoryKey("Directory")
, workDirectoryKey("WorkDirectory")
{
}

XnatBrowserSettings* XnatBrowserSettings::instance()
{
  return &_instance_;
}

QString XnatBrowserSettings::getDefaultURL()
{
  // initialize output XNAT URL
  QString url;

//    pqSettings* settings = pqApplicationCore::instance()->settings();
//    if ( settings )
//    {
//        settings->beginGroup(xnatBrowserGroup);
//
//        url = settings->value(xnatUrlKey, defaultXnatURL).toString();
//
//        settings->endGroup();
//    }

  return url;
}

void XnatBrowserSettings::setDefaultURL(const QString& url)
{
//    pqSettings* settings = pqApplicationCore::instance()->settings();
//    if ( settings )
//    {
//        settings->beginGroup(xnatBrowserGroup);
//
//        settings->setValue(xnatUrlKey, url);
//
//        settings->endGroup();
//    }
}

QString XnatBrowserSettings::getDefaultUserID()
{
  // initialize output XNAT user identifier
  QString userID;

//    pqSettings* settings = pqApplicationCore::instance()->settings();
//    if ( settings )
//    {
//        settings->beginGroup(xnatBrowserGroup);
//
//        userID = settings->value(xnatUserIdKey, defaultXnatUserID).toString();
//
//        settings->endGroup();
//    }

  return userID;
}

void XnatBrowserSettings::setDefaultUserID(const QString& userID)
{
//    pqSettings* settings = pqApplicationCore::instance()->settings();
//    if ( settings )
//    {
//        settings->beginGroup(xnatBrowserGroup);
//
//        settings->setValue(xnatUserIdKey, userID);
//
//        settings->endGroup();
//    }
}

QString XnatBrowserSettings::getDefaultDirectory()
{
  // initialize output directory name
  QString dir;

//    pqSettings* settings = pqApplicationCore::instance()->settings();
//    if ( settings )
//    {
//        settings->beginGroup(xnatBrowserGroup);
//
//        dir = settings->value(directoryKey, defaultDirectory).toString();
//
//        settings->endGroup();
//    }

  return dir;
}

void XnatBrowserSettings::setDefaultDirectory(const QString& dir)
{
//    pqSettings* settings = pqApplicationCore::instance()->settings();
//    if ( settings )
//    {
//        settings->beginGroup(xnatBrowserGroup);
//
//        settings->setValue(directoryKey, dir);
//
//        settings->endGroup();
//    }
}

QString XnatBrowserSettings::getDefaultWorkDirectory()
{
  // initialize output work directory name
  QString workDir;

//    pqSettings* settings = pqApplicationCore::instance()->settings();
//    if ( settings )
//    {
//        settings->beginGroup(xnatBrowserGroup);
//
//        workDir = settings->value(workDirectoryKey, defaultWorkDirectory).toString();
//
//        settings->endGroup();
//    }

  return workDir;
}

void XnatBrowserSettings::setDefaultWorkDirectory(const QString& workDir)
{
//    pqSettings* settings = pqApplicationCore::instance()->settings();
//    if ( settings )
//    {
//        settings->beginGroup(xnatBrowserGroup);
//
//        settings->setValue(workDirectoryKey, workDir);
//
//        settings->endGroup();
//    }
}

QString XnatBrowserSettings::getWorkSubdirectory()
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

