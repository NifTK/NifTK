#ifndef XnatSettings_h
#define XnatSettings_h

#include <QMap>
#include <QString>

class XnatLoginProfile;

class XnatSettings
{
public:
//  virtual QString getDefaultURL() const = 0;
//  virtual void setDefaultURL(const QString& url) = 0;

//  virtual QString getDefaultUserID() const = 0;
//  virtual void setDefaultUserID(const QString& userID) = 0;

  virtual QString getDefaultDirectory() const = 0;
  virtual void setDefaultDirectory(const QString& dir) = 0;

  virtual QString getDefaultWorkDirectory() const = 0;
  virtual void setDefaultWorkDirectory(const QString& workDir) = 0;

  virtual QString getWorkSubdirectory() const;

  virtual QMap<QString, XnatLoginProfile*> getLoginProfiles() const = 0;
  virtual void setLoginProfiles(QMap<QString, XnatLoginProfile*> loginProfiles) = 0;

  virtual XnatLoginProfile* getLoginProfile(QString profileName) const = 0;
  virtual void setLoginProfile(QString profileName, XnatLoginProfile*) = 0;

  virtual XnatLoginProfile* getDefaultLoginProfile() const = 0;

protected:
  explicit XnatSettings();
  virtual ~XnatSettings();
};

#endif
