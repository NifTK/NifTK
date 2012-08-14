#ifndef XNATUPLOADDIALOG_H
#define XNATUPLOADDIALOG_H

#include <QDialog>
#include <QObject>

class QLabel;

class XnatUploadDialog : public QDialog
{
    Q_OBJECT

    public:
        XnatUploadDialog(QWidget* parent);
        void showUploadStarting();
        void showBytesUploaded(unsigned long numBytes);
        bool wasUploadCanceled();
        void closeEvent(QCloseEvent* event);

    public slots:
        bool close();

    private slots:
        void cancelClicked();

    private:
        bool uploadCanceled;
        QLabel* statusLabel;
};

inline bool XnatUploadDialog::wasUploadCanceled() { return uploadCanceled; }

#endif
