#ifndef XnatRootNode_h
#define XnatRootNode_h

#include "XnatRestWidgetsExports.h"

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatRootNode : public XnatNode
{
public:
  explicit XnatRootNode();
  virtual ~XnatRootNode();

  virtual XnatNode* makeChildNode(int row);
};

#endif
