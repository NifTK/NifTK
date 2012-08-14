#ifndef XnatNameDialog_h
#define XnatNameDialog_h

#include <QDialog>

class QLineEdit;

class XnatNameDialog : public QDialog
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
