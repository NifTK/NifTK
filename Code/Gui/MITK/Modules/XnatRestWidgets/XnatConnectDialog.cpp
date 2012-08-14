#include "XnatConnectDialog.h"

#include <QDialogButtonBox>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>

//#include "XnatBrowserSettings.h"
#include "XnatConnection.h"
#include "XnatException.h"


XnatConnectDialog::XnatConnectDialog(XnatConnectionFactory& f, QWidget* parent)
: QDialog(parent)
, factory(f)
, connection(0)
{
  QLabel* urlLabel = new QLabel(tr("&XNAT URL:"));
  urlEdit = new QLineEdit;
//    urlEdit->setText(XnatBrowserSettings::getDefaultURL());
  urlLabel->setBuddy(urlEdit);

  QLabel* userLabel = new QLabel(tr("&User ID:"));
  userEdit = new QLineEdit;
//    userEdit->setText(XnatBrowserSettings::getDefaultUserID());
  userLabel->setBuddy(userEdit);

  QLabel* passwordLabel = new QLabel(tr("&Password:"));
  passwordEdit = new QLineEdit;
  passwordEdit->setEchoMode(QLineEdit::Password);
  passwordLabel->setBuddy(passwordEdit);

  QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok|
                                                     QDialogButtonBox::Cancel);
  connect(buttonBox, SIGNAL(accepted()), this, SLOT(accept()));
  connect(buttonBox, SIGNAL(rejected()), this, SLOT(reject()));

  QGridLayout* gridLayout = new QGridLayout;
  gridLayout->addWidget(urlLabel, 0, 0);
  gridLayout->addWidget(urlEdit, 0, 1);
  gridLayout->addWidget(userLabel, 1, 0);
  gridLayout->addWidget(userEdit, 1, 1);
  gridLayout->addWidget(passwordLabel, 2, 0);
  gridLayout->addWidget(passwordEdit, 2, 1);
  gridLayout->addWidget(buttonBox, 3, 0, 1, 2);
  setLayout(gridLayout);

  setWindowTitle(tr("Login to XNAT"));
  setFixedHeight(sizeHint().height());
  setMinimumWidth(400);

  QLineEdit* inputEdit = ( urlEdit->text().isEmpty() ) ? urlEdit :
                           ( ( userEdit->text().isEmpty() ) ? userEdit : passwordEdit );
  inputEdit->selectAll();
  inputEdit->setFocus();
}

XnatConnection* XnatConnectDialog::getConnection()
{
  return connection;
}

void XnatConnectDialog::accept()
{
  // get input XNAT URL
  QString url = urlEdit->text();
  if ( url.isEmpty() )
  {
    QMessageBox::warning(this, tr("Missing XNAT URL Error"), tr("Please enter XNAT URL."));
    urlEdit->selectAll();
    urlEdit->setFocus();
    return;
  }

  // get input user ID
  QString user = userEdit->text();
  if ( user.isEmpty() )
  {
    QMessageBox::warning(this, tr("Missing User ID Error"), tr("Please enter user ID."));
    userEdit->selectAll();
    userEdit->setFocus();
    return;
  }

  // get input user password
  QString password = passwordEdit->text();
  if ( password.isEmpty() )
  {
    QMessageBox::warning(this, tr("Missing Password Error"), tr("Please enter password."));
    passwordEdit->selectAll();
    passwordEdit->setFocus();
    return;
  }

  // create XNAT connection
  try
  {
    connection = factory.makeConnection(url.toAscii().constData(), user.toAscii().constData(),
                                        password.toAscii().constData());
  }
  catch (XnatException& e)
  {
    QMessageBox::warning(this, tr("Invalid Login Error"), tr(e.what()));
    urlEdit->selectAll();
    urlEdit->setFocus();
    return;
  }

  // save XNAT URL and user ID as defaults
//    XnatBrowserSettings::setDefaultURL(url);
//    XnatBrowserSettings::setDefaultUserID(user);

  QDialog::accept();
}
