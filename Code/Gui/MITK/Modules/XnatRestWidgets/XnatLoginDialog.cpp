#include "XnatLoginDialog.h"

#include <QMessageBox>

#include "XnatConnection.h"
#include "XnatException.h"
#include "XnatSettings.h"

class XnatLoginDialogPrivate
{
public:
  XnatLoginDialogPrivate(XnatConnectionFactory& f)
  : factory(f)
  {
  }

  XnatSettings* settings;

  XnatConnectionFactory& factory;
  XnatConnection* connection;
};

XnatLoginDialog::XnatLoginDialog(XnatConnectionFactory& f, QWidget* parent, Qt::WindowFlags flags)
: QDialog(parent, flags)
, ui(0)
, d_ptr(new XnatLoginDialogPrivate(f))
{
  Q_D(XnatLoginDialog);

  // initialize data members
  d->settings = 0;
  d->connection = 0;

  if (!ui)
  {
    // Create UI
    ui = new Ui::XnatLoginDialog();
    ui->setupUi(this);

    // Create connections after setting defaults, so you don't trigger stuff when setting defaults.
    createConnections();
  }
}

XnatLoginDialog::~XnatLoginDialog()
{
  Q_D(XnatLoginDialog);

  if (ui)
  {
    delete ui;
  }
}

void XnatLoginDialog::createConnections()
{
}

XnatSettings* XnatLoginDialog::settings() const
{
  Q_D(const XnatLoginDialog);

  return d->settings;
}

void XnatLoginDialog::setSettings(XnatSettings* settings)
{
  Q_D(XnatLoginDialog);
  d->settings = settings;
  ui->edtServerUri->setText(settings->getDefaultURL());
  ui->edtUserName->setText(settings->getDefaultUserID());
}

XnatConnection* XnatLoginDialog::getConnection()
{
  Q_D(XnatLoginDialog);
  return d->connection;
}

void XnatLoginDialog::accept()
{
  Q_D(XnatLoginDialog);

  // get input XNAT URL
  QString url = ui->edtServerUri->text();
  if ( url.isEmpty() )
  {
    QMessageBox::warning(this, tr("Missing XNAT URL Error"), tr("Please enter XNAT URL."));
    ui->edtServerUri->selectAll();
    ui->edtServerUri->setFocus();
    return;
  }

  // get input user ID
  QString user = ui->edtUserName->text();
  if ( user.isEmpty() )
  {
    QMessageBox::warning(this, tr("Missing User ID Error"), tr("Please enter user ID."));
    ui->edtUserName->selectAll();
    ui->edtUserName->setFocus();
    return;
  }

  // get input user password
  QString password = ui->edtPassword->text();
  if ( password.isEmpty() )
  {
    QMessageBox::warning(this, tr("Missing Password Error"), tr("Please enter password."));
    ui->edtPassword->selectAll();
    ui->edtPassword->setFocus();
    return;
  }

  // create XNAT connection
  try
  {
    d->connection = d->factory.makeConnection(url.toAscii().constData(), user.toAscii().constData(),
                                        password.toAscii().constData());
  }
  catch (XnatException& e)
  {
    QMessageBox::warning(this, tr("Invalid Login Error"), tr(e.what()));
    ui->edtServerUri->selectAll();
    ui->edtServerUri->setFocus();
    return;
  }

  // save XNAT URL and user ID as defaults
  if (d->settings)
  {
    d->settings->setDefaultURL(url);
    d->settings->setDefaultUserID(user);
  }

  QDialog::accept();
}
