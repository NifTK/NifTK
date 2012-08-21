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
