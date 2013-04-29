/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatConnectionFactory_h
#define XnatConnectionFactory_h

#include "XnatRestWidgetsExports.h"

class XnatConnection;

class XnatRestWidgets_EXPORT XnatConnectionFactory
{
public:
  static XnatConnectionFactory& instance();
  ~XnatConnectionFactory();

  XnatConnection* makeConnection(const char* url, const char* user, const char* password);

private:
  XnatConnectionFactory();
  XnatConnectionFactory& operator=(XnatConnectionFactory& f);
  XnatConnectionFactory(const XnatConnectionFactory& f);

  void testConnection(XnatConnection* conn);
};

#endif
