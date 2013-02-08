/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

extern "C"
{
#include <XnatRest.h>
}

#include "XnatConnectionFactory.h"

#include "XnatConnection.h"
#include "XnatException.h"


// XnatConnectionFactory class

XnatConnectionFactory::XnatConnectionFactory()
{
  // initialize XnatRest
  XnatRestStatus status = initXnatRest();
  if ( status != XNATREST_OK )
  {
    // handle error
    throw XnatException(status);
  }
}

XnatConnectionFactory::~XnatConnectionFactory()
{
  // clean up XnatRest
  cleanupXnatRest();
}

XnatConnectionFactory& XnatConnectionFactory::instance()
{
  static XnatConnectionFactory connectionFactory;
  return connectionFactory;
}

XnatConnection* XnatConnectionFactory::makeConnection(const char* url, const char* user, const char* password)
{
  XnatRestStatus status;

  // set URL address for XNAT web site
  status = setXnatRestUrl(url);
  if ( status != XNATREST_OK )
  {
    // handle error
    throw XnatException(status);
  }

  // set user ID and password for XNAT web site
  status = setXnatRestUser(user, password);
  if ( status != XNATREST_OK )
  {
    // handle error
    throw XnatException(status);
  }

  // create XNAT connection
  XnatConnection* conn = new XnatConnection;

  // test XNAT connection
  try
  {
    testConnection(conn);
  }
  catch (XnatException& e)
  {
    delete conn;
    throw;
  }

  // return XNAT connection
  return conn;
}

void XnatConnectionFactory::testConnection(XnatConnection* conn)
{
  // test connection by retrieving project names from XNAT
  XnatNode* rootNode = NULL;
  try
  {
    // create XNAT root node
    rootNode = conn->getRoot();
    // create project nodes
    rootNode->makeChildNode(0);
    delete rootNode;  // which also deletes childNode
  }
  catch (XnatException& e)
  {
    if (rootNode != NULL)
    {
      delete rootNode;
    }
    throw;
  }
}
