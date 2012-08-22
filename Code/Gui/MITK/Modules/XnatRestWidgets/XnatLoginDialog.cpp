#include "XnatLoginDialog.h"

#include <mitkLogMacros.h>

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

  bool dirty;

//  XnatLoginProfile* currentProfile;
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
//  d->currentProfile = 0;
  d->dirty = false;

  if (!ui)
  {
    // Create UI
    ui = new Ui::XnatLoginDialog();
    ui->setupUi(this);

    ui->lstProfiles->setModel(&d->model);
    ui->lstProfiles->setEditTriggers(QAbstractItemView::NoEditTriggers);

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
  connect(ui->lstProfiles->selectionModel(), SIGNAL(currentChanged(const QModelIndex&, const QModelIndex&)),
          this, SLOT(onCurrentProfileChanged(const QModelIndex&, const QModelIndex&)));
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

  d->model.setStringList(d->profileNames);

  XnatLoginProfile* defaultProfile = d->settings->getDefaultLoginProfile();
  MITK_INFO << "XnatLoginDialog::setSettings(XnatSettings* settings): default profile: " << defaultProfile;

  if (defaultProfile)
  {
    MITK_INFO << "XnatLoginDialog::setSettings(XnatSettings* settings): default profile name: " << defaultProfile->name().toStdString();
    int profileNumber = d->profileNames.indexOf(defaultProfile->name());
    QModelIndex index = d->model.index(profileNumber);
    if (index.isValid())
    {
      MITK_INFO << "XnatLoginDialog::setSettings(XnatSettings* settings): valid index";
//      emit ui->lstProfiles->clicked(index);
      ui->lstProfiles->setCurrentIndex(index);
      on_lstProfiles_clicked(index);
    }
    else
    {
      MITK_INFO << "XnatLoginDialog::setSettings(XnatSettings* settings): invalid index";
    }
//    ui->edtProfileName->setText(defaultProfile->name());
//    ui->edtServerUri->setText(defaultProfile->serverUri());
//    ui->edtUserName->setText(defaultProfile->userName());
  }
  else
  {
    MITK_INFO << "XnatLoginDialog::setSettings(XnatSettings* settings): no default profile";
  }
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

  QDialog::accept();
}

void XnatLoginDialog::on_lstProfiles_clicked(const QModelIndex& index)
{
}

void XnatLoginDialog::onCurrentProfileChanged(const QModelIndex& current, const QModelIndex& previous)
{
  Q_D(XnatLoginDialog);
  MITK_INFO << "XnatLoginDialog::onCurrentProfileChanged(const QModelIndex& index)";
  QString currentProfileName = d->model.data(current, 0).toString();
  QString previousProfileName = d->model.data(previous, 0).toString();
  MITK_INFO << "XnatLoginDialog::onCurrentProfileChanged(const QModelIndex& index) current: " << currentProfileName.toStdString();
  MITK_INFO << "XnatLoginDialog::onCurrentProfileChanged(const QModelIndex& index) previous: " << previousProfileName.toStdString();

  QString profileName = d->model.data(current, 0).toString();
  XnatLoginProfile* profile = d->profiles[profileName];
  ui->edtProfileName->setText(profile->name());
  ui->edtServerUri->setText(profile->serverUri());
  ui->edtUserName->setText(profile->userName());
  ui->edtPassword->setText(profile->password());
  ui->cbxDefaultProfile->setChecked(profile->isDefault());
//  d->currentProfile = profile;
}

void XnatLoginDialog::on_btnSave_clicked()
{
  Q_D(XnatLoginDialog);

  QString profileName = ui->edtProfileName->text();
  QString serverUri = ui->edtServerUri->text();
  QString userName = ui->edtUserName->text();
  QString password = ui->edtPassword->text();
  bool default_ = ui->cbxDefaultProfile->isChecked();

  XnatLoginProfile* profile = d->profiles[profileName];
  if (!profile)
  {
    profile = new XnatLoginProfile();
    d->profiles[profileName] = profile;
  }

  // If the profile is to be default then remove the default flag from the other profiles.
  // This code assumes that the newly created profiles are not default.
  if (default_ && !profile->isDefault())
  {
    foreach (XnatLoginProfile* p, d->profiles.values())
    {
      if (p->name() != profileName && p->isDefault())
      {
        p->setDefault(false);
      }
    }
  }

  profile->setName(profileName);
  profile->setServerUri(serverUri);
  profile->setUserName(userName);
  profile->setPassword(password);
  profile->setDefault(default_);

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

void XnatLoginDialog::on_cbxDefaultProfile_toggled(bool checked)
{
//  Q_D(XnatLoginDialog);
}

void XnatLoginDialog::askConfirmationToSaveProfile()
{
  QString question = "Do you want to save these settings?";
  bool ok = QMessageBox::question(this, "", question, QMessageBox::Save | QMessageBox::Discard,
                                QMessageBox::Save);
}
