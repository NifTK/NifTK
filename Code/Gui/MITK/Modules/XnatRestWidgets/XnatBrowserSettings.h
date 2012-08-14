#ifndef XnatBrowserSettings_h
#define XnatBrowserSettings_h

#include <QString>


class XnatBrowserSettings
{
public:
  static XnatBrowserSettings* instance();

  QString getDefaultURL();
  void setDefaultURL(const QString& url);

  QString getDefaultUserID();
  void setDefaultUserID(const QString& userID);

  QString getDefaultDirectory();
  void setDefaultDirectory(const QString& dir);

  QString getDefaultWorkDirectory();
  void setDefaultWorkDirectory(const QString& workDir);

  QString getWorkSubdirectory();

protected:
  XnatBrowserSettings();

private:
  const QString defaultXnatURL;
  const QString defaultXnatUserID;
  const QString defaultDirectory;
  const QString defaultWorkDirectory;
  const QString xnatBrowserGroup;
  const QString xnatUrlKey;
  const QString xnatUserIdKey;
  const QString directoryKey;
  const QString workDirectoryKey;

  static XnatBrowserSettings _instance_;
};

#endif
