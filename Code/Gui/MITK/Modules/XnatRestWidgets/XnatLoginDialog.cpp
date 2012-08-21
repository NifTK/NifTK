#include "XnatLoginDialog.h"

#include <QMap>
#include <QMessageBox>
#include <QStringListModel>

#include "XnatConnection.h"
#include "XnatException.h"
#include "XnatLoginProfile.h"
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

  QMap<QString, XnatLoginProfile*> profiles;

  QStringListModel model;
  QStringList profileNames;
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

    ui->lstProfiles->setModel(&d->model);

    // Create connections after setting defaults, so you don't trigger stuff when setting defaults.
    createConnections();
  }
}

XnatLoginDialog::~XnatLoginDialog()
{
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
  d->profiles = d->settings->getLoginProfiles();

  d->profileNames = d->profiles.keys();

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

void XnatLoginDialog::on_lstProfiles_clicked(const QModelIndex& index)
{
  Q_D(XnatLoginDialog);
  QString profileName = d->model.data(index, 0).toString();
  XnatLoginProfile* profile = d->profiles[profileName];
  ui->edtProfileName->setText(profile->name());
  ui->edtServerUri->setText(profile->serverUri());
  ui->edtUserName->setText(profile->userName());
  ui->edtPassword->setText(profile->password());
}

void XnatLoginDialog::on_btnSave_clicked()
{
  Q_D(XnatLoginDialog);

  QString profileName = ui->edtProfileName->text();
  QString serverUri = ui->edtServerUri->text();
  QString userName = ui->edtUserName->text();
  QString password = ui->edtPassword->text();

  XnatLoginProfile* profile = d->profiles[profileName];
  if (!profile)
  {
	profile = new XnatLoginProfile();
	d->profiles[profileName] = profile;
  }
  profile->setName(profileName);
  profile->setServerUri(serverUri);
  profile->setUserName(userName);
  profile->setPassword(password);

  d->settings->setLoginProfiles(d->profiles);

  d->profileNames.push_back(profileName);
  d->model.setStringList(d->profileNames);
}

void XnatLoginDialog::on_btnDelete_clicked()
{
  Q_D(XnatLoginDialog);

  QString profileName = ui->edtProfileName->text();

  d->profileNames.removeOne(profileName);
  d->model.setStringList(d->profileNames);

  d->profiles.remove(profileName);

  d->settings->setLoginProfiles(d->profiles);
}
