#include "XnatConnectionDialog.h"

#include "ui_XnatConnectionDialog.h"

class XnatConnectionDialogPrivate
{
public:
  QString serverUri;
  QString username;
  QString password;
};

XnatConnectionDialog::XnatConnectionDialog(QWidget *parent)
: QDialog(parent)
, ui(new Ui::XnatConnectionDialogClass)
, d_ptr(new XnatConnectionDialogPrivate())
{
  Q_D(XnatConnectionDialog);

  d->serverUri = "http://localhost:8080/xnat";
  d->username = "espakm";
  d->password = "demo";

  ui->setupUi(this);

  ui->serverUriLineEdit->setText(d->serverUri);
  ui->usernameLineEdit->setText(d->username);
  ui->passwordLineEdit->setText(d->password);
}

XnatConnectionDialog::~XnatConnectionDialog()
{
  delete ui;
}

void XnatConnectionDialog::accept()
{
  Q_D(XnatConnectionDialog);
  d->serverUri = ui->serverUriLineEdit->text();
  d->username = ui->usernameLineEdit->text();
  d->password = ui->passwordLineEdit->text();
  QDialog::accept();
}

QString XnatConnectionDialog::serverUri() const
{
  Q_D(const XnatConnectionDialog);
  return d->serverUri;
}

QString XnatConnectionDialog::username() const
{
  Q_D(const XnatConnectionDialog);
  return d->username;
}

QString XnatConnectionDialog::password() const
{
  Q_D(const XnatConnectionDialog);
  return d->password;
}
