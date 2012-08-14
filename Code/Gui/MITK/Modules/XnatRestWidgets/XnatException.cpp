#include "XnatException.h"

extern "C"
{
#include <XnatRest.h>
}

XnatException::XnatException(const XnatRestStatus& status)
: message(getXnatRestStatusMsg(status))
{
}

const char* XnatException::what() const throw()
{
  return message;
}
