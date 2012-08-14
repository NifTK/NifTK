#ifndef XNATEXCEPTION_H
#define XNATEXCEPTION_H

extern "C" {
#include "XnatRest.h"
}


class XnatException
{
    public:
        XnatException(XnatRestStatus status);
        const char* what();

    private:
        const char *message;
};

inline XnatException::XnatException(XnatRestStatus status) : message(getXnatRestStatusMsg(status)) {}

inline const char* XnatException::what() { return message; }

#endif
