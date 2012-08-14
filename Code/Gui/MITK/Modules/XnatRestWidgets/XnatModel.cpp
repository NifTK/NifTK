#include "XnatModel.h"

#include "XnatException.h"
#include "XnatNodeProperties.h"


const int XnatModel::ModifiableChildKind = Qt::UserRole + 1;
const int XnatModel::ModifiableParentName = Qt::UserRole + 2;

XnatModel::XnatModel(XnatNode* rootNode)
: root(rootNode)
{
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
    return QVariant(int(Qt::AlignTop | Qt::AlignLeft));
  }
  else if ( !index.isValid() )
  {
    return QVariant();
  }

  XnatNode* node = (XnatNode*) index.internalPointer();
  if ( role == Qt::DisplayRole )
  {
    return QVariant(node->getChildName(index.row()));
  }
  else if ( role == Qt::ToolTipRole )
  {
    const char* toolTip = node->getKind();
    if ( toolTip != NULL )
    {
      return QVariant(toolTip);
    }
  }
  else if ( role == Qt::UserRole )
  {
    XnatNodeProperties nodeProperties(index.row(), node);
    return QVariant(nodeProperties.getBitArray());
  }
  else if ( role == ModifiableChildKind )
  {
    return QVariant(node->getModifiableChildKind(index.row()));
  }
  else if ( role == ModifiableParentName )
  {
    return QVariant(node->getModifiableParentName(index.row()));
  }

  return QVariant();
}

QString XnatModel::name(const QModelIndex& index) const
{
  return data(index, Qt::DisplayRole).toString();
}

QModelIndex XnatModel::parent(const QModelIndex& child) const
{
  if ( !child.isValid() )
  {
    return QModelIndex();
  }

  XnatNode* node = (XnatNode*) child.internalPointer();
  XnatNode* parent = node->getParentNode();
  if ( parent == NULL )
  {
    return QModelIndex();
  }
  return createIndex(node->getRowInParent(), 0, (void*) parent);
}

QModelIndex XnatModel::index(int row, int column, const QModelIndex& parent) const
{
  if ( column != 0 )
  {
    return QModelIndex();
  }
  if ( !parent.isValid() )
  {
    if ( ( row < root->getNumChildren() ) && ( row >= 0 ) )
    {
      return createIndex(row, column, (void*) root);
    }
    else
    {
      return QModelIndex();
    }
  }
  XnatNode* node = (XnatNode*) parent.internalPointer();
  XnatNode* childNode = node->getChildNode(parent.row());
  if ( childNode == NULL )
  {
    return QModelIndex();
  }
  if ( ( row < childNode->getNumChildren() ) && ( row >= 0 ) )
  {
    return createIndex(row, column, (void*) childNode);
  }
  return QModelIndex();
}

int XnatModel::rowCount(const QModelIndex& parent) const
{
  if ( !parent.isValid() )
  {
    return ((int) (root->getNumChildren()));
  }
  XnatNode* node = (XnatNode*) parent.internalPointer();
  XnatNode* childNode = node->getChildNode(parent.row());
  if ( childNode == NULL )
  {
    return 0;
  }
  return ((int) (childNode->getNumChildren()));
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
  if ( parent.isValid() )
  {
    XnatNode* node = (XnatNode*) parent.internalPointer();
    XnatNode* childNode = node->getChildNode(parent.row());
    if ( childNode == NULL )
    {
      try
      {
        XnatNode* tempNode = node->makeChildNode(parent.row());
        if ( tempNode != NULL )
        {
          if ( tempNode->getNumChildren() > 0 )
          {
            fetchFlag = true;
          }
          delete tempNode;
        }
      }
      catch (XnatException& e)
      {
        return false;
      }
    }
  }
  return fetchFlag;
}

void XnatModel::fetchMore(const QModelIndex& parent)
{
  if ( parent.isValid() )
  {
    XnatNode* node = (XnatNode*) parent.internalPointer();
    XnatNode* childNode = node->getChildNode(parent.row());
    if ( childNode == NULL )
    {
      try
      {
        XnatNode* tempNode = node->makeChildNode(parent.row());
        if ( tempNode != NULL )
        {
          size_t numChildren = tempNode->getNumChildren();
          if ( numChildren > 0 )
          {
            beginInsertRows(parent, 0, ((int)(numChildren - 1)));
            node->addChildNode(parent.row(), tempNode);
            endInsertRows();
          }
          else
          {
            delete tempNode;
          }
        }
      }
      catch (XnatException& e)
      {
        return;
      }
    }
  }
}

bool XnatModel::removeAllRows(const QModelIndex& parent)
{
  // do nothing for root node
  if ( !parent.isValid() )
  {
    return false;
  }

  XnatNode* node = (XnatNode*) parent.internalPointer();
  XnatNode* childNode = node->getChildNode(parent.row());
  if ( childNode == NULL )
  {
    return false;
  }
  size_t numChildren = childNode->getNumChildren();
  if ( numChildren > 0 )
  {
    beginRemoveRows(parent, 0, ((int) (numChildren - 1)));
    node->removeChildNode(parent.row());
    endRemoveRows();
  }
  else
  {
    node->removeChildNode(parent.row());
  }
  return true;
}

void XnatModel::downloadFile(const QModelIndex& index, const QString& zipFilename)
{
  if ( !index.isValid() )
  {
    return;
  }

  XnatNode* node = (XnatNode*) index.internalPointer();
  node->download(index.row(), zipFilename.toAscii().constData());
}

void XnatModel::downloadFileGroup(const QModelIndex& index, const QString& zipFilename)
{
  if ( !index.isValid() )
  {
    return;
  }

  XnatNode* node = (XnatNode*) index.internalPointer();
  node->downloadAllFiles(index.row(), zipFilename.toAscii().constData());
}

void XnatModel::uploadFile(const QModelIndex& index, const QString& zipFilename)
{
  if ( !index.isValid() )
  {
    return;
  }

  XnatNode* node = (XnatNode*) index.internalPointer();
  node->upload(index.row(), zipFilename.toAscii().constData());
}

void XnatModel::addEntry(const QModelIndex& index, const QString& name)
{
  if ( !index.isValid() )
  {
    return;
  }

  XnatNode* node = (XnatNode*) index.internalPointer();
  node->add(index.row(), name.toAscii().constData());
}

void XnatModel::removeEntry(const QModelIndex& index)
{
  if ( !index.isValid() )
  {
    return;
  }

  XnatNode* node = (XnatNode*) index.internalPointer();
  node->remove(index.row());
}
