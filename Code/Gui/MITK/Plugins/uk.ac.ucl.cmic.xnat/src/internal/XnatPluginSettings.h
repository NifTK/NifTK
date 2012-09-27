/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: me $

 Original author   : m.espak@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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

  virtual void removeLoginProfile(QString profileName);

  XnatLoginProfile* getDefaultLoginProfile() const;

private:
  berry::IPreferences::Pointer preferences;
};

#endif
