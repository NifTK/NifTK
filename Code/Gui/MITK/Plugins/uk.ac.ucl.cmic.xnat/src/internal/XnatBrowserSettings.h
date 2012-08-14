#ifndef XNATBROWSERSETTINGS_H
#define XNATBROWSERSETTINGS_H

#include <QString>


class XnatBrowserSettings
{
public:
  static QString getDefaultURL();
  static void setDefaultURL(const QString& url);

  static QString getDefaultUserID();
  static void setDefaultUserID(const QString& userID);

  static QString getDefaultDirectory();
  static void setDefaultDirectory(const QString& dir);

  static QString getDefaultWorkDirectory();
  static void setDefaultWorkDirectory(const QString& workDir);

  static QString getWorkSubdirectory();

private:
  XnatBrowserSettings();

  static const QString defaultXnatURL;
  static const QString defaultXnatUserID;
  static const QString defaultDirectory;
  static const QString defaultWorkDirectory;
  static const QString xnatBrowserGroup;
  static const QString xnatUrlKey;
  static const QString xnatUserIdKey;
  static const QString directoryKey;
  static const QString workDirectoryKey;
};

#endif
