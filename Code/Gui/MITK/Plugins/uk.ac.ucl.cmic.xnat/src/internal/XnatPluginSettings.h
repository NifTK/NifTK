/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatPluginSettings_h
#define XnatPluginSettings_h

#include <berryIPreferences.h>

#include <ctkXnatSettings.h>

#include <QString>
#include <QMap>

class XnatPluginSettings : public ctkXnatSettings
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

  virtual QMap<QString, ctkXnatLoginProfile*> getLoginProfiles() const;
  virtual void setLoginProfiles(QMap<QString, ctkXnatLoginProfile*> loginProfiles);

  virtual ctkXnatLoginProfile* getLoginProfile(QString profileName) const;
  virtual void setLoginProfile(QString profileName, ctkXnatLoginProfile*);

  virtual void removeLoginProfile(QString profileName);

  ctkXnatLoginProfile* getDefaultLoginProfile() const;

private:
  berry::IPreferences::Pointer preferences;
};

#endif
