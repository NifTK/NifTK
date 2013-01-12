/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatBrowserWidget_h
#define XnatBrowserWidget_h

#include <QDir>
#include <QStringList>
#include <QWidget>

#include "ui_XnatBrowserWidget.h"

#include <mitkDataStorage.h>

class QModelIndex;
class XnatBrowserWidgetPrivate;
class XnatSettings;
class XnatObject;

class XnatBrowserWidget : public QWidget
{
  Q_OBJECT

public:
  explicit XnatBrowserWidget(QWidget* parent = 0, Qt::WindowFlags flags = 0);
  virtual ~XnatBrowserWidget();

  mitk::DataStorage::Pointer dataStorage() const;
  void setDataStorage(mitk::DataStorage::Pointer dataStorage);

  XnatSettings* settings() const;
  void setSettings(XnatSettings* settings);

private slots:
  void loginXnat();
  void importFile();
  void importFiles();
  void setButtonEnabled(const QModelIndex& index);
  void setSaveAndUploadButtonEnabled();
  void showContextMenu(const QPoint&);

private:
  void createConnections();
  void collectImageFiles(const QDir& tempWorkDirectory, QStringList& fileList);

  /// \brief All the controls for the main view part.
  Ui::XnatBrowserWidget* ui;

  /// \brief d pointer of the pimpl pattern
  QScopedPointer<XnatBrowserWidgetPrivate> d_ptr;

  Q_DECLARE_PRIVATE(XnatBrowserWidget);
  Q_DISABLE_COPY(XnatBrowserWidget);
};

#endif
