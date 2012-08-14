#ifndef XNATFILENAMETYPEDIALOG_H
#define XNATFILENAMETYPEDIALOG_H

#include <QDialog>
#include <QObject>

class QLineEdit;
class QComboBox;


class XnatFileNameTypeDialog : public QDialog
{
  Q_OBJECT

public:
  XnatFileNameTypeDialog (const QString& filters, QWidget* parent);
  QString getFilename();

private slots:
  void accept();

private:
  QString inputFname;
  QStringList filterList;

  QLineEdit* nameEdit;
  QComboBox* typeComboBox;

  QStringList makeFilterList(const QString& filters);
  QStringList getWildCardListFromFilter(const QString& filter);
  QString fixFilePathExtension(const QString& filePath, const QString& filter);
};


inline QString XnatFileNameTypeDialog::getFilename() { return inputFname; }

#endif
