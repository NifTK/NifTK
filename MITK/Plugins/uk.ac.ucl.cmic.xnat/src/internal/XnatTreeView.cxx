#include "XnatTreeView.h"

#include <QItemSelectionModel>
#include <QModelIndex>
#include <QMessageBox>

#include <ctkXnatSession.h>
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

void XnatTreeView::initialize(ctkXnatSession* connection)
{
  ctkXnatTreeModel* xnatModel = this->xnatModel();
  if (xnatModel)
  {
    delete xnatModel;
  }
  xnatModel = new ctkXnatTreeModel();
  ctkXnatDataModel* server = connection->dataModel();
  xnatModel->addDataModel(server);

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

const ctkXnatObject* XnatTreeView::currentObject()
{
  return this->xnatObject(this->selectionModel()->currentIndex());
}

const ctkXnatObject* XnatTreeView::xnatObject(const QModelIndex& index)
{
  return this->xnatModel()->xnatObject(index);
}

void XnatTreeView::refreshRows()
{
  QModelIndex index = this->selectionModel()->currentIndex();
  ctkXnatTreeModel* model = (ctkXnatTreeModel*) this->model();
  model->removeAllRows(index);
  model->fetchMore(index);
}
