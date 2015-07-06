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

namespace berry
{
class IPreferences;
template <class T> class SmartPointer;
}

class XnatPluginSettings : public ctkXnatSettings
{
public:
  XnatPluginSettings(berry::IPreferences::Pointer preferences);

  virtual QString defaultURL() const;
  virtual void setDefaultURL(const QString& url);

  virtual QString defaultUserID() const;
  virtual void setDefaultUserID(const QString& userID);

  virtual QString defaultDirectory() const;
  virtual void setDefaultDirectory(const QString& dir);

  virtual QString defaultWorkDirectory() const;
  virtual void setDefaultWorkDirectory(const QString& workDir);

  virtual QMap<QString, ctkXnatLoginProfile*> loginProfiles() const;
  virtual void setLoginProfiles(QMap<QString, ctkXnatLoginProfile*> loginProfiles);

  virtual ctkXnatLoginProfile* loginProfile(QString profileName) const;
  virtual void setLoginProfile(QString profileName, ctkXnatLoginProfile*);

  virtual void removeLoginProfile(QString profileName);

  ctkXnatLoginProfile* defaultLoginProfile() const;

private:
  berry::SmartPointer<berry::IPreferences> preferences;
};

#endif
