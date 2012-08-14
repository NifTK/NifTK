/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-19 17:52:47 +0100 (Tue, 19 Jul 2011) $
 Revision          : $Revision: 6804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "XnatBrowserView.h"

#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>

#include <QtNetwork>
#include <QNetworkAccessManager>
#include <QLabel>
#include <QVBoxLayout>
#include <QUrl>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QImageReader>
#include <QMessageBox>
#include <QScriptEngine>
#include <QStringBuilder>

#include "XnatBrowserWidget.h"
#include "XnatConnectionDialog.h"
#include "XnatPluginPreferencePage.h"
#include "XnatPluginSettings.h"

const std::string XnatBrowserView::VIEW_ID = "uk.ac.ucl.cmic.imagelookuptables";

class XnatBrowserViewPrivate
{
public:

  QNetworkAccessManager* networkAccessManager;
};

XnatBrowserView::XnatBrowserView()
: m_Controls(0)
, m_Parent(0)
, d_ptr(new XnatBrowserViewPrivate())
{
  Q_D(XnatBrowserView);
  d->networkAccessManager = new QNetworkAccessManager(this);
}

XnatBrowserView::~XnatBrowserView()
{
  if (m_Controls)
  {
    delete m_Controls;
  }
}

void XnatBrowserView::CreateQtPartControl(QWidget *parent)
{
  // setup the basic GUI of this view
  m_Parent = parent;

  if (!m_Controls)
  {
    // Create UI
//    m_Controls = new Ui::XnatBrowserView();
    XnatBrowserWidget* xnatBrowserWidget = new XnatBrowserWidget(parent);
    xnatBrowserWidget->setSettings(new XnatPluginSettings(GetPreferences()));
    QVBoxLayout* layout = new QVBoxLayout();
    layout->addWidget(xnatBrowserWidget);
    parent->setLayout(layout);
//    m_Controls->setupUi(parent);

    // Create connections after setting defaults, so you don't trigger stuff when setting defaults.
//    CreateConnections();
  }
}

void XnatBrowserView::CreateConnections()
{
//  Q_D(XnatBrowserView);
//
//  connect(m_Controls->connectButton, SIGNAL(clicked()), this, SLOT(on_connectButton_clicked()));
//  connect(d->networkAccessManager, SIGNAL(finished(QNetworkReply*)), this, SLOT(onNetworkReply(QNetworkReply*)));
////  connect(d->networkAccessManager, SIGNAL(authenticationRequired(QNetworkReply*, QAuthenticator*)),
////          this, SLOT(onAuthenticationRequired(QNetworkReply*, QAuthenticator*)));
//#ifndef QT_NO_OPENSSL
//  connect(d->networkAccessManager, SIGNAL(sslErrors(QNetworkReply*, QList<QSslError>)),
//          this, SLOT(onSslErrors(QNetworkReply*, QList<QSslError>)));
//#endif
}

void XnatBrowserView::SetFocus()
{
//  m_Controls->connectButton->setFocus();
}

void XnatBrowserView::on_connectButton_clicked()
{
  XnatConnectionDialog* connectionDialog = new XnatConnectionDialog(m_Parent);

  if (connectionDialog->exec() == QDialog::Accepted)
  {
    QString serverUri = connectionDialog->serverUri();
    QString username = connectionDialog->username();
    QString password = connectionDialog->password();
    MITK_INFO << "server uri: " << serverUri.toStdString();
    MITK_INFO << "username: " << username.toStdString();
    MITK_INFO << "password: " << password.toStdString();

    serverUri.append("/data/archive/projects");

    QUrl projectsQuery(serverUri);

    QNetworkRequest request;
    request.setRawHeader("Authorization", "Basic " +
        QByteArray(QString("%1:%2").arg(username).arg(password).toAscii()).toBase64());
    request.setUrl(projectsQuery);

    MITK_INFO << "projectsQuery: " << projectsQuery.toString().toStdString();

//    QNetworkReply* reply = d->networkAccessManager->get(request);
    // NOTE: Store QNetworkReply pointer (maybe into caller).
    // When this HTTP request is finished you will receive this same
    // QNetworkReply as response parameter.
    // By the QNetworkReply pointer you can identify request and response.
  }

  delete connectionDialog;
}

void XnatBrowserView::onNetworkReply(QNetworkReply* reply)
{
  // Reading attributes of the reply
  // e.g. the HTTP status code
  QVariant statusCodeV = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute);
  // Or the target URL if it was a redirect:
  QVariant redirectionTargetUrl = reply->attribute(QNetworkRequest::RedirectionTargetAttribute);
  // see CS001432 on how to handle this

  // no error received?
  if (reply->error() == QNetworkReply::NoError)
  {
    // read data from QNetworkReply here

    // Example 1: Creating QImage from the reply
//    QImageReader imageReader(reply);
//    QImage pic = imageReader.read();

    // Example 2: Reading bytes form the reply
    QByteArray bytes = reply->readAll();  // bytes

    QString result = QString::fromUtf8(bytes); // string
    MITK_INFO << "response: " << result.toStdString();
    QScriptValue sc;
    QScriptEngine engine;
    sc = engine.evaluate(result); // In new versions it may need to look like engine.evaluate("(" + QString(result) + ")");

//    if (sc.property("result").isArray())
//    {
//      QStringList items;
//      qScriptValueToSequence(sc.property("result"), items);
//
//      foreach (QString str, items)
//      {
//        qDebug("value %s",str.toStdString().c_str());
//      }
//    }
  }
  // Some http error received
  else
  {
    // handle errors here
    QMessageBox::critical(m_Parent, "Network error", reply->errorString());

  }

  // We receive ownership of the reply object
  // and therefore need to handle deletion.
  reply->deleteLater();
}

void XnatBrowserView::onAuthenticationRequired(QNetworkReply* reply, QAuthenticator* authenticator)
{
  MITK_INFO << "void XnatBrowserView::onAuthenticationRequired(QNetworkReply* reply, QAuthenticator* authenticator)";
  authenticator->setUser(QString("espakm"));
  authenticator->setPassword(QString("demo"));
}

#ifndef QT_NO_OPENSSL
void XnatBrowserView::onSslErrors(QNetworkReply* reply, const QList<QSslError>& errors)
{
  QString errorString;
  foreach (const QSslError& error, errors)
  {
    if (!errorString.isEmpty())
    {
      errorString.append(", ");
    }
    errorString.append(error.errorString());
  }

  if (QMessageBox::warning(m_Parent, tr("HTTP"),
          tr("One or more SSL errors has occurred: %1").arg(errorString),
          QMessageBox::Ignore | QMessageBox::Abort) == QMessageBox::Ignore)
  {
    reply->ignoreSslErrors();
  }
}
#endif
