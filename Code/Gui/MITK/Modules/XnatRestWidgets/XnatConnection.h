#ifndef XNATCONNECTION_H
#define XNATCONNECTION_H

#include "XnatNode.h"


class XnatConnection
{
    public:
        XnatNode* getRoot();
};

class XnatConnectionFactory
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
