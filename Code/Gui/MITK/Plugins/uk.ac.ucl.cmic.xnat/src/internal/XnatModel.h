/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatModel_h
#define XnatModel_h

#include <QAbstractItemModel>

class ctkXnatConnection;
class ctkXnatObject;
class ctkXnatServer;

class XnatModel : public QAbstractItemModel
{
  Q_OBJECT

public:
  explicit XnatModel(ctkXnatConnection* connection);
  virtual ~XnatModel();

  virtual QVariant data(const QModelIndex& index, int role) const;
  virtual QModelIndex parent(const QModelIndex& child) const;
  virtual QModelIndex index(int row, int column, const QModelIndex& parent) const;
  virtual int rowCount(const QModelIndex& parent) const;
  virtual int columnCount(const QModelIndex& parent) const;
  virtual bool hasChildren(const QModelIndex& parent) const;
  virtual bool canFetchMore(const QModelIndex& parent) const;
  virtual void fetchMore(const QModelIndex& parent);
  bool removeAllRows(const QModelIndex& parent);

  void downloadFile(const QModelIndex& index, const QString& zipFilename);
  void uploadFile(const QModelIndex& index, const QString& zipFilename);
  void addEntry(const QModelIndex& index, const QString& name);
  void removeEntry(const QModelIndex& index);

  static const int ModifiableChildKind;
  static const int ModifiableParentName;

private:
  ctkXnatConnection* connection;
  ctkXnatObject* root;
  ctkXnatServer* server;
};

#endif
