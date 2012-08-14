#ifndef XnatTreeView_h
#define XnatTreeView_h

#include <QTreeView>

class XnatNode;

class XnatTreeView : public QTreeView
{
  Q_OBJECT

public:
  explicit XnatTreeView(QWidget* parent = 0);
  virtual ~XnatTreeView();

  void initialize(XnatNode* rootNode);
};

#endif /* XnatTreeView_h */
