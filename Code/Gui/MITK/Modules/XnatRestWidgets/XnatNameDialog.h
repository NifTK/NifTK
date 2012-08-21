#ifndef XnatNameDialog_h
#define XnatNameDialog_h

#include "XnatRestWidgetsExports.h"

#include <QDialog>

class QLineEdit;

class XnatRestWidgets_EXPORT XnatNameDialog : public QDialog
{
  Q_OBJECT

public:
  XnatNameDialog(QWidget* p, const QString& kind, const QString& parentName);
  const QString getNewName();

private slots:
  void accept();

private:
  QLineEdit* nameEdit;
  QString newName;
};

#endif
