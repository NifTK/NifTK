#ifndef XnatConnectDialog_h
#define XnatConnectDialog_h

#include <QDialog>

#include "XnatConnectionFactory.h"

class QLineEdit;
class XnatConnection;
class XnatSettings;

class XnatConnectDialog : public QDialog
{
  Q_OBJECT

public:
  explicit XnatConnectDialog(XnatConnectionFactory& factory, QWidget* parent);
  virtual ~XnatConnectDialog();

  XnatConnection* getConnection();

  void setSettings(XnatSettings* settings);

private slots:
  void accept();

private:
  XnatSettings* settings;

  XnatConnectionFactory& factory;
  XnatConnection* connection;

  QLineEdit* urlEdit;
  QLineEdit* userEdit;
  QLineEdit* passwordEdit;
};

#endif
