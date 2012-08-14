#ifndef XnatException_h
#define XnatException_h

#include <exception>

#include <XnatRestStatus.h>

class XnatException : public std::exception
{
public:
  XnatException(const XnatRestStatus& status);

  virtual const char* what() const throw();

private:
  const char* message;
};

#endif
