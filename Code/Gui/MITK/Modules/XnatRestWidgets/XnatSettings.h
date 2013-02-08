/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatSettings_h
#define XnatSettings_h

#include "XnatRestWidgetsExports.h"

#include <QMap>
#include <QString>

class XnatLoginProfile;

class XnatRestWidgets_EXPORT XnatSettings
{
public:
  virtual QString getDefaultDirectory() const = 0;
  virtual void setDefaultDirectory(const QString& dir) = 0;

  virtual QString getDefaultWorkDirectory() const = 0;
  virtual void setDefaultWorkDirectory(const QString& workDir) = 0;

  virtual QString getWorkSubdirectory() const;

  virtual QMap<QString, XnatLoginProfile*> getLoginProfiles() const = 0;
  virtual void setLoginProfiles(QMap<QString, XnatLoginProfile*> loginProfiles) = 0;

  virtual XnatLoginProfile* getLoginProfile(QString profileName) const = 0;
  virtual void setLoginProfile(QString profileName, XnatLoginProfile*) = 0;

  virtual void removeLoginProfile(QString profileName) = 0;

  virtual XnatLoginProfile* getDefaultLoginProfile() const = 0;

protected:
  explicit XnatSettings();
  virtual ~XnatSettings();
};

#endif
