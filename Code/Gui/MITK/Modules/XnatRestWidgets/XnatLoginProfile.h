#ifndef XnatLoginProfile_h
#define XnatLoginProfile_h

#include <QString>

class XnatLoginProfile
{
public:
  explicit XnatLoginProfile();
  virtual ~XnatLoginProfile();

  QString name() const;
  void setName(const QString& profileName);

  QString serverUri() const;
  void setServerUri(const QString& serverUri);

  QString userName() const;
  void setUserName(const QString& userName);

  QString password() const;
  void setPassword(const QString& password);

private:
  QString m_name;
  QString m_serverUri;
  QString m_userName;
  QString m_password;
};

#endif
