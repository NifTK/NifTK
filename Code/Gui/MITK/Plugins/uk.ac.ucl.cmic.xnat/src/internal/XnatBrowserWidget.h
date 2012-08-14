#ifndef XnatBrowserWidget_h
#define XnatBrowserWidget_h

#include <QWidget>

#include "ui_XnatBrowserWidget.h"

class QModelIndex;
class XnatBrowserWidgetPrivate;
class XnatSettings;
class XnatNode;

class XnatBrowserWidget : public QWidget
{
  Q_OBJECT

public:
  explicit XnatBrowserWidget(QWidget* parent = 0, Qt::WindowFlags flags = 0);
  virtual ~XnatBrowserWidget();

  XnatSettings* settings() const;
  void setSettings(XnatSettings* settings);

  bool startFileUpload(const QString& zipFilename);
  bool startFileDownload(const QString& zipFilename);
  bool startFileGroupDownload(const QString& zipFilename);

  void refreshRows();

private slots:
  void loginXnat();
  void downloadFile();
  void downloadAllFiles();
  void downloadAndOpenFile();
  void createNewRow();
  void deleteCurrentRow();
  void setButtonEnabled(const QModelIndex& index);
  void setSaveDataAndUploadButtonEnabled();
  void showContextMenu(const QPoint&);

private:
  void createConnections();
  void initializeTreeView(XnatNode* rootNode);

  /// \brief All the controls for the main view part.
  Ui::XnatBrowserWidget* ui;

  /// \brief d pointer of the pimpl pattern
  QScopedPointer<XnatBrowserWidgetPrivate> d_ptr;

  Q_DECLARE_PRIVATE(XnatBrowserWidget);
  Q_DISABLE_COPY(XnatBrowserWidget);
};

#endif
