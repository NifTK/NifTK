#ifndef CONNECTIONDIALOG_H
#define CONNECTIONDIALOG_H

#include <QtGui/QDialog>

class XnatConnectionDialogPrivate;

namespace Ui {
  class XnatConnectionDialogClass;
}

class XnatConnectionDialog : public QDialog
{
  Q_OBJECT

public:
  XnatConnectionDialog(QWidget *parent = 0);
  virtual ~XnatConnectionDialog();

  QString serverUri() const;
  QString username() const;
  QString password() const;

public slots:

  virtual void accept();

private:
  Ui::XnatConnectionDialogClass* ui;

  QScopedPointer<XnatConnectionDialogPrivate> d_ptr;

  Q_DECLARE_PRIVATE(XnatConnectionDialog);
  Q_DISABLE_COPY(XnatConnectionDialog);
};

#endif // CONNECTIONDIALOG_H
