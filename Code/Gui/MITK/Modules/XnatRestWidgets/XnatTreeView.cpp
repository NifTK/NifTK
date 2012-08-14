#include "XnatTreeView.h"

#include <QItemSelectionModel>
#include <QModelIndex>

#include "XnatModel.h"
#include "XnatNode.h"
#include "XnatNodeActivity.h"

XnatTreeView::XnatTreeView(QWidget* parent)
: QTreeView(parent)
{
  setSelectionBehavior(QTreeView::SelectItems);
  setUniformRowHeights(true);
  setHeaderHidden(true);

  initialize(new XnatNode(XnatEmptyNodeActivity::instance()));
}

XnatTreeView::~XnatTreeView()
{
}

void XnatTreeView::initialize(XnatNode* rootNode)
{
  XnatModel* oldModel = (XnatModel*) this->model();
  QItemSelectionModel* oldSelectionModel = this->selectionModel();
  this->setModel(new XnatModel(rootNode));
  if (oldModel)
  {
    delete oldModel;
  }
  if (oldSelectionModel)
  {
    delete oldSelectionModel;
  }
  this->setExpanded(QModelIndex(), false);
}
