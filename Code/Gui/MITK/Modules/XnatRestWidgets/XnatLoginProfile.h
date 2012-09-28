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

#ifndef XnatLoginProfile_h
#define XnatLoginProfile_h

#include "XnatRestWidgetsExports.h"

#include <QString>

class XnatRestWidgets_EXPORT XnatLoginProfile
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
