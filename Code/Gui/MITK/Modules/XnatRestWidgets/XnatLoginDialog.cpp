#include "XnatLoginDialog.h"

#include <mitkLogMacros.h>

#include <QMap>
#include <QMessageBox>
#include <QStringListModel>
#include <QListView>

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

  QModelIndex currentIndex;
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
  d->dirty = false;

  if (!ui)
  {
    // Create UI
    ui = new Ui::XnatLoginDialog();
    ui->setupUi(this);

    ui->lstProfiles->setModel(&d->model);
    ui->lstProfiles->setSelectionMode(QAbstractItemView::SingleSelection);
    ui->lstProfiles->setSelectionRectVisible(false);
    ui->lstProfiles->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui->btnSave->setEnabled(false);

    // Create connections after setting defaults, so you don't trigger stuff when setting defaults.
    createConnections();
  }
}

XnatLoginDialog::~XnatLoginDialog()
{
  Q_D(XnatLoginDialog);

  foreach (XnatLoginProfile* profile, d->profiles)
  {
    delete profile;
  }

  if (ui)
  {
    delete ui;
  }
}

void XnatLoginDialog::createConnections()
{
  connect(ui->edtProfileName, SIGNAL(textChanged(const QString&)), this, SLOT(onFieldChanged()));
  connect(ui->edtServerUri, SIGNAL(textChanged(const QString&)), this, SLOT(onFieldChanged()));
  connect(ui->edtUserName, SIGNAL(textChanged(const QString&)), this, SLOT(onFieldChanged()));
  // Password change is not listened to.
//  connect(ui->edtPassword, SIGNAL(textChanged(const QString&)), this, SLOT(onFieldChanged()));
  connect(ui->cbxDefaultProfile, SIGNAL(toggled(bool)), this, SLOT(onFieldChanged()));
  connect(ui->lstProfiles->selectionModel(), SIGNAL(currentChanged(const QModelIndex&, const QModelIndex&)),
      this, SLOT(onCurrentProfileChanged(const QModelIndex&)));
//  connect(ui->lstProfiles, SIGNAL(clicked(const QModelIndex&)),
//      this, SLOT(onCurrentProfileChanged(const QModelIndex&)));
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
  d->profileNames.sort();
  d->model.setStringList(d->profileNames);

  XnatLoginProfile* defaultProfile = d->settings->getDefaultLoginProfile();

  if (defaultProfile)
  {
    int profileNumber = d->profileNames.indexOf(defaultProfile->name());
    QModelIndex index = d->model.index(profileNumber);
    if (index.isValid())
    {
      ui->lstProfiles->setCurrentIndex(index);
//      on_lstProfiles_clicked(index);
    }
    ui->edtPassword->setFocus();
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

  QString url = ui->edtServerUri->text();
  if ( url.isEmpty() )
  {
    QMessageBox::warning(this, tr("Missing XNAT server URI"), tr("Please enter XNAT server URI."));
    ui->edtServerUri->selectAll();
    ui->edtServerUri->setFocus();
    return;
  }

  QString userName = ui->edtUserName->text();
  if ( userName.isEmpty() )
  {
    QMessageBox::warning(this, tr("Missing user name"), tr("Please enter user name."));
    ui->edtUserName->selectAll();
    ui->edtUserName->setFocus();
    return;
  }

  if (d->dirty)
  {
    const QString& profileName = ui->edtProfileName->text();
    if (askToSaveProfile(profileName))
    {
      saveProfile(profileName);
    }
  }

  QString password = ui->edtPassword->text();

  // create XNAT connection
  try
  {
    d->connection = d->factory.makeConnection(url.toAscii().constData(), userName.toAscii().constData(),
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

//void XnatLoginDialog::on_lstProfiles_clicked(const QModelIndex& currentIndex)
//{
//  QModelIndex oldIndex = currentIndex;
//}

void XnatLoginDialog::onCurrentProfileChanged(const QModelIndex& currentIndex)
{
  Q_D(XnatLoginDialog);
  QString newProfileName = d->profileNames[currentIndex.row()];

  if (d->dirty)
  {
    QString profileName = ui->edtProfileName->text();
    if (askToSaveProfile(profileName))
    {
      saveProfile(profileName);
    }
  }

  if (!currentIndex.isValid())
  {
    return;
  }

  if (d->currentIndex == currentIndex) {
      return;
  }

  XnatLoginProfile* profile = d->profiles[newProfileName];
  if (profile)
  {
    loadProfile(*profile);

    d->dirty = false;
    ui->btnSave->setEnabled(false);
    ui->btnDelete->setEnabled(true);

    // The profile index has changed if a profile was saved a few lines above.
    // Does not deselect the old line!
//    ui->lstProfiles->selectionModel()->select(oldIndex, QItemSelectionModel::Clear);
//    ui->lstProfiles->selectionModel()->select(oldIndex, QItemSelectionModel::Deselect);
    ui->lstProfiles->setCurrentIndex(currentIndex);

    d->currentIndex = currentIndex;
  }
}

bool XnatLoginDialog::askToSaveProfile(const QString& profileName)
{
  QString question = QString(
      "You have not saved the changes of the %1 profile.\n"
      "Do you want to save them now?").arg(profileName);
  QMessageBox::StandardButton answer = QMessageBox::question(this, "", question, QMessageBox::Yes | QMessageBox::No,
                                QMessageBox::Yes);

  return answer == QMessageBox::Yes;
}

void XnatLoginDialog::saveProfile(const QString& profileName)
{
  Q_D(XnatLoginDialog);

//  XnatLoginProfile newProfile;

  XnatLoginProfile* profile = d->profiles[profileName];
  bool oldProfileWasDefault = profile && profile->isDefault();
  if (!profile)
  {
    profile = new XnatLoginProfile();
    d->profiles[profileName] = profile;
    int profileNumber = d->profileNames.size();

    // Insertion into the profile name list and the listView (ascending order)
    int idx = 0;
    while (idx < profileNumber && QString::localeAwareCompare(profileName, d->profileNames[idx]) > 0)
    {
      ++idx;
    }
    d->profileNames.insert(idx, profileName);
    d->model.insertRow(idx);
    d->model.setData(d->model.index(idx), profileName);
  }

  storeProfile(*profile);

  // If the profile is to be default then remove the default flag from the other profiles.
  // This code assumes that the newly created profiles are not default.
  if (profile->isDefault() && !oldProfileWasDefault)
  {
    MITK_INFO << "save profile: if (default_ && !profile->isDefault()) : true";
    foreach (XnatLoginProfile* otherProfile, d->profiles.values())
    {
      const QString& otherProfileName = otherProfile->name();
      MITK_INFO << "save profile: other profile : " << otherProfileName.toStdString();
      if (otherProfileName != profileName && otherProfile->isDefault())
      {
        MITK_INFO << "save profile: other profile different: ";
        otherProfile->setDefault(false);
        d->settings->setLoginProfile(otherProfileName, otherProfile);
      }
    }
  }

  d->settings->setLoginProfile(profileName, profile);
  d->dirty = false;
  ui->btnSave->setEnabled(false);
}

void XnatLoginDialog::on_btnSave_clicked()
{
  Q_D(XnatLoginDialog);
  QString editedProfileName = ui->edtProfileName->text();

  QModelIndex currentIndex = ui->lstProfiles->currentIndex();
  int selectedProfileNumber = currentIndex.row();
  QString selectedProfileName = d->profileNames[selectedProfileNumber];

  saveProfile(editedProfileName);

  if (editedProfileName != selectedProfileName)
  {
    int editedProfileNumber = d->profileNames.indexOf(editedProfileName);
    QModelIndex editedProfileIndex = d->model.index(editedProfileNumber, 0);
    ui->lstProfiles->setCurrentIndex(editedProfileIndex);
  }
}

void XnatLoginDialog::blockSignalsOfFields(bool value)
{
  ui->edtProfileName->blockSignals(value);
  ui->edtServerUri->blockSignals(value);
  ui->edtUserName->blockSignals(value);
  ui->edtPassword->blockSignals(value);
  ui->cbxDefaultProfile->blockSignals(value);
}

void XnatLoginDialog::loadProfile(const XnatLoginProfile& profile)
{
  blockSignalsOfFields(true);

  ui->edtProfileName->setText(profile.name());
  ui->edtServerUri->setText(profile.serverUri());
  ui->edtUserName->setText(profile.userName());
  ui->edtPassword->setText(profile.password());
  ui->cbxDefaultProfile->setChecked(profile.isDefault());

  blockSignalsOfFields(false);
}

void XnatLoginDialog::storeProfile(XnatLoginProfile& profile)
{
  profile.setName(ui->edtProfileName->text());
  profile.setServerUri(ui->edtServerUri->text());
  profile.setUserName(ui->edtUserName->text());
  profile.setPassword(ui->edtPassword->text());
  profile.setDefault(ui->cbxDefaultProfile->isChecked());
}

void XnatLoginDialog::on_btnDelete_clicked()
{
  Q_D(XnatLoginDialog);


  QString profileName = ui->edtProfileName->text();

  int idx = d->profileNames.indexOf(profileName);
  d->profileNames.removeAt(idx);
  d->model.removeRow(idx);
//  d->model.setData(d->model.index(idx), profileName);
//  d->model.setStringList(d->profileNames);

  delete d->profiles.take(profileName);

  ui->lstProfiles->clearSelection();
  d->currentIndex = QModelIndex();
  ui->btnDelete->setEnabled(false);

  loadProfile();

  d->settings->removeLoginProfile(profileName);
}

void XnatLoginDialog::on_edtProfileName_textChanged(const QString& text)
{
  ui->lstProfiles->clearSelection();
  ui->btnDelete->setEnabled(false);
}

void XnatLoginDialog::onFieldChanged()
{
  Q_D(XnatLoginDialog);
  d->dirty = true;
  ui->btnSave->setEnabled(true);
}
