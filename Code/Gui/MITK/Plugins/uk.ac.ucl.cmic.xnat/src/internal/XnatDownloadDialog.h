#ifndef XnatDownloadDialog_h
#define XnatDownloadDialog_h

#include <QDialog>

class QCloseEvent;
class QLabel;

class XnatDownloadDialog : public QDialog
{
  Q_OBJECT

public:
  XnatDownloadDialog(QWidget* parent);
  void showBytesDownloaded(unsigned long numBytes);
  void showUnzipInProgress();
  bool wasDownloadCanceled();
  void closeEvent(QCloseEvent* event);

public slots:
  bool close();

private slots:
  void cancelClicked();

private:
  bool downloadCanceled;
  QLabel* statusLabel;
};

#endif
