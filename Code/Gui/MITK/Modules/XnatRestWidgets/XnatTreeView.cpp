#include "XnatTreeView.h"

#include <QItemSelectionModel>
#include <QModelIndex>
#include <QMessageBox>

#include "XnatException.h"
#include "XnatModel.h"
#include "XnatNameDialog.h"
#include "XnatNode.h"
#include "XnatNodeActivity.h"

class XnatTreeViewPrivate
{
public:
//  XnatModel* xnatModel;
};

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
  // clean up models in tree view
  XnatModel* xnatModel = this->xnatModel();
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

void XnatTreeView::initialize(XnatNode* rootNode)
{
  XnatModel* xnatModel = this->xnatModel();
  if (xnatModel)
  {
    delete xnatModel;
  }
  xnatModel = new XnatModel(rootNode);
  this->setModel(xnatModel);

//  QItemSelectionModel* selectionModel = this->selectionModel();
//  if (selectionModel)
//  {
//    delete selectionModel;
//  }
  this->setExpanded(QModelIndex(), false);
}

XnatModel* XnatTreeView::xnatModel()
{
  return dynamic_cast<XnatModel*>(model());
}

XnatNodeProperties XnatTreeView::currentNodeProperties()
{
  return nodeProperties(this->selectionModel()->currentIndex());
}

XnatNodeProperties XnatTreeView::nodeProperties(const QModelIndex& index)
{
  return XnatNodeProperties(this->model()->data(index, Qt::UserRole).toBitArray());
}

void XnatTreeView::refreshRows()

{
  QModelIndex index = this->selectionModel()->currentIndex();
  XnatModel* model = (XnatModel*) this->model();
  model->removeAllRows(index);
  model->fetchMore(index);
}

void XnatTreeView::createNewRow()
{
  QModelIndex index = this->selectionModel()->currentIndex();
  XnatModel* model = this->xnatModel();

  // get kind of new entry, e.g., reconstruction or resource
  QString childKind = model->data(index, XnatModel::ModifiableChildKind).toString();
  if ( childKind.isEmpty() )
  {
    QMessageBox::warning(this, tr("Create New Error"), tr("Unknown child kind"));
    return;
  }

  // get parent name, e.g., experiment name for new reconstruction, or
  //                        reconstruction name for new resource
  QString parentName = model->data(index, XnatModel::ModifiableParentName).toString();
  if ( parentName.isEmpty() )
  {
    QMessageBox::warning(this, tr("Create New Error"), tr("Unknown parent name"));
    return;
  }

  // get name of new child in parent from user, e.g.,
  //             name of new reconstruction in experiment, or
  //             name of new resource in reconstruction
  XnatNameDialog nameDialog(this, childKind, parentName);
  if ( nameDialog.exec() )
  {
    QString name = nameDialog.getNewName();

    try
    {
      // create new child in parent, e.g., new reconstruction in experiment, or
      //                                   new resource in reconstruction
      model->addEntry(index, name);

      // refresh display
      model->removeAllRows(index);
      model->fetchMore(index);
    }
    catch (XnatException& e)
    {
      QMessageBox::warning(this, tr("Create New Error"), tr(e.what()));
    }
  }
}

void XnatTreeView::deleteCurrentRow()
{
  // get name in row to be deleted
  QModelIndex index = this->selectionModel()->currentIndex();
  deleteRow(index);
}

void XnatTreeView::deleteRow(const QModelIndex& index)
{
  // get name in row to be deleted
  XnatModel* model = this->xnatModel();
//  QString name = model->name(index);
  QString name = model->data(index, Qt::DisplayRole).toString();

  // ask user to confirm deletion
  int buttonPressed = QMessageBox::question(this, tr("Confirm Deletion"), tr("Delete %1 ?").arg(name),
                                            QMessageBox::Yes | QMessageBox::No);

  if ( buttonPressed == QMessageBox::Yes )
  {
    try
    {
      // delete row
      QModelIndex parent = model->parent(index);
      model->removeEntry(index);

      // refresh display
      model->removeAllRows(parent);
      model->fetchMore(parent);
    }
    catch (XnatException& e)
    {
      QMessageBox::warning(this, tr("Delete Error"), tr(e.what()));
    }
  }
}
