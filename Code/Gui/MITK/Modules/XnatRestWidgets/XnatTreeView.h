#ifndef XnatTreeView_h
#define XnatTreeView_h

#include <QTreeView>
#include <QModelIndex>

#include "XnatNodeProperties.h"

class XnatModel;
class XnatNode;
class XnatTreeViewPrivate;

class XnatTreeView : public QTreeView
{
  Q_OBJECT

public:
  explicit XnatTreeView(QWidget* parent = 0);
  virtual ~XnatTreeView();

  void initialize(XnatNode* rootNode);

  XnatModel* xnatModel();

  XnatNodeProperties nodeProperties(const QModelIndex& index);
  XnatNodeProperties currentNodeProperties();

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
