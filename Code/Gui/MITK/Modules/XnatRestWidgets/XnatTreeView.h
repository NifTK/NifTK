/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatTreeView_h
#define XnatTreeView_h

#include "XnatRestWidgetsExports.h"

#include <QTreeView>
#include <QModelIndex>

class XnatModel;
class XnatNode;
class XnatTreeViewPrivate;

class XnatRestWidgets_EXPORT XnatTreeView : public QTreeView
{
  Q_OBJECT

public:
  explicit XnatTreeView(QWidget* parent = 0);
  virtual ~XnatTreeView();

  void initialize(XnatNode* rootNode);

  XnatModel* xnatModel();

  const XnatNode* node(const QModelIndex& index);
  const XnatNode* currentNode();

public slots:
  void refreshRows();
  void createNewRow();
  void deleteRow(const QModelIndex& index);
  void deleteCurrentRow();

private:
  /// \brief d pointer of the pimpl pattern
  QScopedPointer<XnatTreeViewPrivate> d_ptr;

  Q_DECLARE_PRIVATE(XnatTreeView);
};

#endif /* XnatTreeView_h */
