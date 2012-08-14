#ifndef XnatPluginSettings_h
#define XnatPluginSettings_h

#include <QString>

#include <berryIPreferences.h>

#include <XnatBrowserSettings.h>

class XnatPluginSettings : public XnatBrowserSettings
{
public:
  XnatPluginSettings(berry::IPreferences::Pointer preferences);

  QString getDefaultURL();
  void setDefaultURL(const QString& url);

  QString getDefaultUserID();
  void setDefaultUserID(const QString& userID);

  QString getDefaultDirectory();
  void setDefaultDirectory(const QString& dir);

  QString getDefaultWorkDirectory();
  void setDefaultWorkDirectory(const QString& workDir);

  QString getWorkSubdirectory();

private:
  berry::IPreferences::Pointer preferences;
};

#endif
