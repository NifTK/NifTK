#ifndef XnatPluginSettings_h
#define XnatPluginSettings_h

#include <QString>

#include <berryIPreferences.h>

#include <XnatSettings.h>

class XnatPluginSettings : public XnatSettings
{
public:
  XnatPluginSettings(berry::IPreferences::Pointer preferences);

  virtual QString getDefaultURL() const;
  virtual void setDefaultURL(const QString& url);

  virtual QString getDefaultUserID() const;
  virtual void setDefaultUserID(const QString& userID);

  virtual QString getDefaultDirectory() const;
  virtual void setDefaultDirectory(const QString& dir);

  virtual QString getDefaultWorkDirectory() const;
  virtual void setDefaultWorkDirectory(const QString& workDir);

private:
  berry::IPreferences::Pointer preferences;
};

#endif
