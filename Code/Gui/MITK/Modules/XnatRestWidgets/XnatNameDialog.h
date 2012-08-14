#ifndef XNATNAMEDIALOG_H
#define XNATNAMEDIALOG_H

#include <QDialog>
#include <QObject>

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


inline const QString XnatNameDialog::getNewName() { return newName; }

#endif
