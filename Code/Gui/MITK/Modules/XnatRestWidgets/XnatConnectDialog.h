#ifndef XNATCONNECTDIALOG_H
#define XNATCONNECTDIALOG_H

#include <QDialog>

#include "XnatConnectionFactory.h"

class QLineEdit;
class XnatConnection;

class XnatConnectDialog : public QDialog
{
  Q_OBJECT

public:
  XnatConnectDialog(XnatConnectionFactory& factory, QWidget* parent);
  XnatConnection* getConnection();

private slots:
  void accept();

private:
  XnatConnectionFactory& factory;
  XnatConnection* connection;

  QLineEdit* urlEdit;
  QLineEdit* userEdit;
  QLineEdit* passwordEdit;
};

#endif
