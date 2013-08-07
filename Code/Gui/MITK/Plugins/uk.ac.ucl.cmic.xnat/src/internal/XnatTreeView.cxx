#include "XnatTreeView.h"

#include <QItemSelectionModel>
#include <QModelIndex>
#include <QMessageBox>

#include <ctkXnatConnection.h>
#include <ctkXnatException.h>
#include <ctkXnatObject.h>
#include <ctkXnatTreeModel.h>

class XnatTreeViewPrivate
{
public:
//  ctkXnatTreeModel* xnatModel;
};

XnatTreeView::XnatTreeView(QWidget* parent)
: QTreeView(parent)
{
  setSelectionBehavior(QTreeView::SelectItems);
  setUniformRowHeights(true);
  setHeaderHidden(true);
}

XnatTreeView::~XnatTreeView()
{
  // clean up models in tree view
  ctkXnatTreeModel* xnatModel = this->xnatModel();
  if (xnatModel)
  {
    delete xnatModel;
  }

//  QItemSelectionModel* selectionModel = this->selectionModel();
//  if (selectionModel)
//  {
//    delete selectionModel;
//  }
}

void XnatTreeView::initialize(ctkXnatConnection* connection)
{
  ctkXnatTreeModel* xnatModel = this->xnatModel();
  if (xnatModel)
  {
    delete xnatModel;
  }
  xnatModel = new ctkXnatTreeModel();
  ctkXnatServer::Pointer server = connection->server();
  xnatModel->addServer (server);

  this->setModel(xnatModel);

//  QItemSelectionModel* selectionModel = this->selectionModel();
//  if (selectionModel)
//  {
//    delete selectionModel;
//  }
  this->setExpanded(QModelIndex(), false);
}

ctkXnatTreeModel* XnatTreeView::xnatModel()
{
  return dynamic_cast<ctkXnatTreeModel*>(model());
}

const ctkXnatObject::Pointer XnatTreeView::currentObject()
{
  return getObject (this->selectionModel()->currentIndex());
}

const ctkXnatObject::Pointer XnatTreeView::getObject(const QModelIndex& index)
{

  ctkXnatTreeItem* item = static_cast<ctkXnatTreeItem*>(index.internalPointer());
  ctkXnatObject::Pointer xnatObject = item->xnatObject();

  return xnatObject;
}

void XnatTreeView::refreshRows()
{
  QModelIndex index = this->selectionModel()->currentIndex();
  ctkXnatTreeModel* model = (ctkXnatTreeModel*) this->model();
  model->removeAllRows(index);
  model->fetchMore(index);
}
