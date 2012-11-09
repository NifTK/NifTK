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
