#ifndef XnatSettings_h
#define XnatSettings_h

#include "XnatRestWidgetsExports.h"

#include <QString>


class XnatRestWidgets_EXPORT XnatSettings
{
public:
  virtual QString getDefaultURL() const = 0;
  virtual void setDefaultURL(const QString& url) = 0;

  virtual QString getDefaultUserID() const = 0;
  virtual void setDefaultUserID(const QString& userID) = 0;

  virtual QString getDefaultDirectory() const = 0;
  virtual void setDefaultDirectory(const QString& dir) = 0;

  virtual QString getDefaultWorkDirectory() const = 0;
  virtual void setDefaultWorkDirectory(const QString& workDir) = 0;

  virtual QString getWorkSubdirectory() const;

protected:
  explicit XnatSettings();
  virtual ~XnatSettings();
};

#endif
