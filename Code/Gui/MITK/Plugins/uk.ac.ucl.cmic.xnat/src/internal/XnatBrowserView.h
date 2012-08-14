/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-19 12:16:16 +0100 (Tue, 19 Jul 2011) $
 Revision          : $Revision: 6802 $
 Last modified by  : $Author: me $

 Original author   : Miklos Espak <m.espak@ucl.ac.uk>

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef _XNATBROWSERVIEW_H_INCLUDED
#define _XNATBROWSERVIEW_H_INCLUDED

#include "ui_XnatBrowserView.h"

#include <berryQtViewPart.h>
#include <berryIBerryPreferences.h>
#include <QmitkAbstractView.h>
#include <mitkDataNode.h>

class QWidget;
class QNetworkReply;
class QSslError;
class QAuthenticator;

class XnatBrowserViewPrivate;

/**
 * \class XnatBrowserView
 * \brief Provides a simple GUI for browsing XNAT databases
 * \ingroup uk_ac_ucl_cmic_xnat_internal
 */
class XnatBrowserView : public QmitkAbstractView
{

  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  explicit XnatBrowserView();
  virtual ~XnatBrowserView();

  /// \brief Each view for a plugin has its own globally unique ID.
  static const std::string VIEW_ID;

protected:

  /// \brief Called by framework, this method creates all the controls for this view
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Called by framework, sets the focus on a specific widget.
  virtual void SetFocus();

  void BuildPythonQt(QWidget *parent);

private slots:

  void on_connectButton_clicked();
  void onNetworkReply(QNetworkReply* reply);
  void onAuthenticationRequired(QNetworkReply* reply, QAuthenticator* authenticator);
#ifndef QT_NO_OPENSSL
  void onSslErrors(QNetworkReply* reply, const QList<QSslError>& errors);
#endif

private:

  /// \brief Creation of the connections of widgets to slots.
  void CreateConnections();

  /// \brief All the controls for the main view part.
  Ui::XnatBrowserView* m_Controls;

  // Store a reference to the parent widget of this view.
  QWidget *m_Parent;

  /// \brief d pointer of the pimpl pattern
  QScopedPointer<XnatBrowserViewPrivate> d_ptr;

  Q_DECLARE_PRIVATE(XnatBrowserView);
  Q_DISABLE_COPY(XnatBrowserView);
};
#endif // _XNATBROWSERVIEW_H_INCLUDED
