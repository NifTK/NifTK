#ifndef XnatPluginSettings_h
#define XnatPluginSettings_h

#include <QString>
#include <QMap>

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

  virtual QMap<QString, XnatLoginProfile*> getLoginProfiles() const;
  virtual void setLoginProfiles(QMap<QString, XnatLoginProfile*> loginProfiles);

  virtual XnatLoginProfile* getLoginProfile(QString profileName) const;
  virtual void setLoginProfile(QString profileName, XnatLoginProfile*);

private:
  berry::IPreferences::Pointer preferences;
};

#endif
