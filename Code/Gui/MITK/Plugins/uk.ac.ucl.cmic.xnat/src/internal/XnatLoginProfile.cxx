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

#include "XnatLoginProfile.h"

XnatLoginProfile::XnatLoginProfile()
{
  m_default = false;
}

XnatLoginProfile::~XnatLoginProfile()
{
}

QString XnatLoginProfile::name() const
{
  return m_name;
}

void XnatLoginProfile::setName(const QString& name)
{
  m_name = name;
}

QString XnatLoginProfile::serverUri() const
{
  return m_serverUri;
}

void XnatLoginProfile::setServerUri(const QString& serverUri)
{
  m_serverUri = serverUri;
}

QString XnatLoginProfile::userName() const
{
  return m_userName;
}

void XnatLoginProfile::setUserName(const QString& userName)
{
  m_userName = userName;
}

QString XnatLoginProfile::password() const
{
  return m_password;
}

void XnatLoginProfile::setPassword(const QString& password)
{
  m_password = password;
}

bool XnatLoginProfile::isDefault() const
{
  return m_default;
}

void XnatLoginProfile::setDefault(const bool& default_)
{
  m_default = default_;
}
