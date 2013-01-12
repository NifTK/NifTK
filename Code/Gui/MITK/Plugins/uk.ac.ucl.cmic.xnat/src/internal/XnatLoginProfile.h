/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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

  bool isDefault() const;
  void setDefault(const bool& default_);

private:
  QString m_name;
  QString m_serverUri;
  QString m_userName;
  QString m_password;
  bool m_default;
};

#endif
