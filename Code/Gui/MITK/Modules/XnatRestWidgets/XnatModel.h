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

#include "XnatRestWidgetsExports.h"

#include <QAbstractItemModel>

#include "XnatNode.h"


class XnatRestWidgets_EXPORT XnatModel : public QAbstractItemModel
{
  Q_OBJECT

public:
  explicit XnatModel(XnatNode* rootNode);
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

//  inline QVariant name(const QModelIndex& index) const
//  {
//    return data(index, Qt::DisplayRole);
//  }

  void downloadFile(const QModelIndex& index, const QString& zipFilename);
  void uploadFile(const QModelIndex& index, const QString& zipFilename);
  void addEntry(const QModelIndex& index, const QString& name);
  void removeEntry(const QModelIndex& index);

  static const int ModifiableChildKind;
  static const int ModifiableParentName;

private:
  XnatNode* root;
};

#endif
