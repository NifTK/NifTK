#include "XnatModel.h"

#include <ctkXnatConnection.h>
#include <ctkXnatException.h>
#include <ctkXnatObject.h>
#include <ctkXnatServer.h>

#include <QDebug>

const int XnatModel::ModifiableChildKind = Qt::UserRole;
const int XnatModel::ModifiableParentName = Qt::UserRole + 1;

XnatModel::XnatModel(ctkXnatConnection* connection)
: connection(connection)
{
  root = new ctkXnatObject();
  ctkXnatServer* server = new ctkXnatServer();
  root->addChild("XNAT", server);
}

XnatModel::~XnatModel()
{
  delete root;
}

// returns name (project, subject, etc.) for row and column of
//   parent in index if role is Qt::DisplayRole
QVariant XnatModel::data(const QModelIndex& index, int role) const
{
  if ( role == Qt::TextAlignmentRole )
  {
//    qDebug() << "role: text aligment";
    return QVariant(int(Qt::AlignTop | Qt::AlignLeft));
  }
  else if ( !index.isValid() )
  {
//    qDebug() << "role: invalid index";
    return QVariant();
  }

  ctkXnatObject* object = (ctkXnatObject*) index.internalPointer();
  ctkXnatObject* child = object->getChildren()[index.row()];
  if ( role == Qt::DisplayRole )
  {
    return QVariant(object->childName(index.row()));
  }
  else if ( role == Qt::ToolTipRole )
  {
    QString toolTip = object->getKind();
    if ( toolTip != NULL )
    {
      return QVariant(toolTip);
    }
  }
  else if ( role == ModifiableChildKind )
  {
    return QVariant(child->getModifiableChildKind());
  }
  else if ( role == ModifiableParentName )
  {
    return QVariant(child->getModifiableParentName());
  }

  return QVariant();
}

QModelIndex XnatModel::parent(const QModelIndex& child) const
{
  if ( !child.isValid() )
  {
    return QModelIndex();
  }

  ctkXnatObject* object = (ctkXnatObject*) child.internalPointer();
  ctkXnatObject* parent = object->getParent();
  if ( parent == NULL )
  {
    return QModelIndex();
  }
  return createIndex(object->parentIndex(), 0, parent);
}

QModelIndex XnatModel::index(int row, int column, const QModelIndex& parent) const
{
  if (column != 0)
  {
    return QModelIndex();
  }
  if (!parent.isValid())
  {
    if (0 <= row && row < root->getChildren().size())
    {
      return createIndex(row, column, root);
    }
    else
    {
      return QModelIndex();
    }
  }
  ctkXnatObject* object = (ctkXnatObject*) parent.internalPointer();
  ctkXnatObject* child = object->getChildren()[parent.row()];
  if (!child)
  {
    return QModelIndex();
  }
  if (0 <= row && row < child->getChildren().size())
  {
    return createIndex(row, column, (void*) child);
  }
  return QModelIndex();
}

int XnatModel::rowCount(const QModelIndex& parent) const
{
  if ( !parent.isValid() )
  {
    return root->getChildren().size();
  }
  ctkXnatObject* object = (ctkXnatObject*) parent.internalPointer();
  ctkXnatObject* child = object->getChildren()[parent.row()];
  if ( child == NULL )
  {
    return 0;
  }
  return child->getChildren().size();
}

int XnatModel::columnCount(const QModelIndex& parent) const
{
  return 1;
}

// defer request for children until actually needed by QTreeView object
bool XnatModel::hasChildren(const QModelIndex& parent) const
{
  return true;
}

bool XnatModel::canFetchMore(const QModelIndex& parent) const
{
  bool fetchFlag = false;
  if (parent.isValid())
  {
    ctkXnatObject* object = (ctkXnatObject*) parent.internalPointer();
    ctkXnatObject* child = object->getChildren()[parent.row()];
    if (child) {
      return true;
    }
  }
  else
  {
//    qDebug() << "object name: ---";
  }
  return fetchFlag;
}

void XnatModel::fetchMore(const QModelIndex& parent)
{
  if (parent.isValid())
  {
    ctkXnatObject* object = (ctkXnatObject*) parent.internalPointer();
    ctkXnatObject* child = object->getChildren()[parent.row()];
    if (child)
    {
      child->fetch(connection);
    }
  }
}

bool XnatModel::removeAllRows(const QModelIndex& parent)
{
  // do nothing for the root
  if ( !parent.isValid() )
  {
    return false;
  }

  ctkXnatObject* object = (ctkXnatObject*) parent.internalPointer();
  ctkXnatObject* child = object->getChildren()[parent.row()];
  if ( child == NULL )
  {
    return false;
  }
  int childNumber = child->getChildren().size();
  if (0 < childNumber)
  {
    beginRemoveRows(parent, 0, childNumber - 1);
    object->removeChild(parent.row());
    endRemoveRows();
  }
  else
  {
    object->removeChild(parent.row());
  }
  return true;
}

void XnatModel::downloadFile(const QModelIndex& index, const QString& zipFilename)
{
  if ( !index.isValid() )
  {
    return;
  }

  ctkXnatObject* object = (ctkXnatObject*) index.internalPointer();
  ctkXnatObject* child = object->getChildren()[index.row()];
  child->download(connection, zipFilename);
}

void XnatModel::uploadFile(const QModelIndex& index, const QString& zipFilename)
{
  if ( !index.isValid() )
  {
    return;
  }

  ctkXnatObject* object = (ctkXnatObject*) index.internalPointer();
  ctkXnatObject* child = object->getChildren()[index.row()];
  child->upload(connection, zipFilename);
}

void XnatModel::addEntry(const QModelIndex& index, const QString& name)
{
  if ( !index.isValid() )
  {
    return;
  }

  ctkXnatObject* object = (ctkXnatObject*) index.internalPointer();
  ctkXnatObject* child = object->getChildren()[index.row()];
  child->add(connection, name);
}

void XnatModel::removeEntry(const QModelIndex& index)
{
  if ( !index.isValid() )
  {
    return;
  }

  ctkXnatObject* object = (ctkXnatObject*) index.internalPointer();
  ctkXnatObject* child = object->getChildren()[index.row()];
  child->remove(connection);
}
