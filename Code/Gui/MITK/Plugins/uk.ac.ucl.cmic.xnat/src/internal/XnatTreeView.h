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

#include <QTreeView>
#include <QModelIndex>

#include "ctkXnatObject.h"


class ctkXnatConnection;
class ctkXnatTreeModel;
class XnatTreeViewPrivate;

class XnatTreeView : public QTreeView
{
  Q_OBJECT

public:
  explicit XnatTreeView(QWidget* parent = 0);
  virtual ~XnatTreeView();

  void initialize(ctkXnatConnection* connection);

  ctkXnatTreeModel* xnatModel();

  const ctkXnatObject::Pointer getObject(const QModelIndex& index);
  const ctkXnatObject::Pointer currentObject();

public slots:
  void refreshRows();

private:
  /// \brief d pointer of the pimpl pattern
  QScopedPointer<XnatTreeViewPrivate> d_ptr;

  Q_DECLARE_PRIVATE(XnatTreeView);
};

#endif /* XnatTreeView_h */
