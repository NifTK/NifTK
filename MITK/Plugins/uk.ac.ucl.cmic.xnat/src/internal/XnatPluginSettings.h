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

#include <ctkXnatSettings.h>

#include <QString>
#include <QMap>

#include <berryIPreferences.h>


class XnatPluginSettings : public ctkXnatSettings
{
public:
  XnatPluginSettings(berry::IPreferences::Pointer preferences);

  virtual QString defaultURL() const;
  virtual void setDefaultURL(const QString& url);

  virtual QString defaultUserID() const;
  virtual void setDefaultUserID(const QString& userID);

  virtual QString defaultDirectory() const override;
  virtual void setDefaultDirectory(const QString& dir) override;

  virtual QString defaultWorkDirectory() const override;
  virtual void setDefaultWorkDirectory(const QString& workDir) override;

  virtual QMap<QString, ctkXnatLoginProfile*> loginProfiles() const override;
  virtual void setLoginProfiles(QMap<QString, ctkXnatLoginProfile*> loginProfiles) override;

  virtual ctkXnatLoginProfile* loginProfile(QString profileName) const override;
  virtual void setLoginProfile(QString profileName, ctkXnatLoginProfile*) override;

  virtual void removeLoginProfile(QString profileName) override;

  ctkXnatLoginProfile* defaultLoginProfile() const override;

private:
  berry::IPreferences::Pointer preferences;
};

#endif
