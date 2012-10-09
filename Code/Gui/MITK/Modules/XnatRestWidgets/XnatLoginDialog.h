/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: me $

 Original author   : m.espak@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef XnatLoginDialog_h
#define XnatLoginDialog_h

#include "XnatRestWidgetsExports.h"

#include <QDialog>

#include "ui_XnatLoginDialog.h"

#include "XnatConnectionFactory.h"
#include "XnatLoginProfile.h"

class XnatConnection;
class XnatLoginDialogPrivate;
class XnatSettings;

class XnatRestWidgets_EXPORT XnatLoginDialog : public QDialog
{
  Q_OBJECT

public:
  explicit XnatLoginDialog(XnatConnectionFactory& f, QWidget* parent = 0, Qt::WindowFlags flags = 0);
  virtual ~XnatLoginDialog();

  XnatSettings* settings() const;
  void setSettings(XnatSettings* settings);

  XnatConnection* getConnection();

  virtual void accept();

private slots:

  void on_btnSave_clicked();
  void on_btnDelete_clicked();
  void on_edtProfileName_textChanged(const QString& text);
  void onFieldChanged();
  void onCurrentProfileChanged(const QModelIndex& current);
  void resetLstProfilesCurrentIndex();

private:
  void createConnections();
  void blockSignalsOfFields(bool value);

  void saveProfile(const QString& profileName);
  bool askToSaveProfile(const QString& profileName);
  void loadProfile(const XnatLoginProfile& profile = XnatLoginProfile());
  void storeProfile(XnatLoginProfile& profile);

  /// \brief All the controls for the main view part.
  Ui::XnatLoginDialog* ui;

  /// \brief d pointer of the pimpl pattern
  QScopedPointer<XnatLoginDialogPrivate> d_ptr;

  Q_DECLARE_PRIVATE(XnatLoginDialog);
  Q_DISABLE_COPY(XnatLoginDialog);
};

#endif
