#ifndef XnatLoginDialog_h
#define XnatLoginDialog_h

#include <QDialog>

#include "ui_XnatLoginDialog.h"

#include "XnatConnectionFactory.h"

class XnatConnection;
class XnatLoginDialogPrivate;
class XnatSettings;

class XnatLoginDialog : public QDialog
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
  void onCurrentProfileChanged(const QModelIndex& current, const QModelIndex& previous);
  void onFieldChanged();

private:
  void createConnections();
  void blockSignalsOfFields(bool value);

  void saveProfile(const QString& profileName);
  bool askToSaveProfile(const QString& profileName);

  /// \brief All the controls for the main view part.
  Ui::XnatLoginDialog* ui;

  /// \brief d pointer of the pimpl pattern
  QScopedPointer<XnatLoginDialogPrivate> d_ptr;

  Q_DECLARE_PRIVATE(XnatLoginDialog);
  Q_DISABLE_COPY(XnatLoginDialog);
};

#endif
