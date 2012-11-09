#ifndef XnatEmptyNode_h
#define XnatEmptyNode_h

#include "XnatRestWidgetsExports.h"

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatEmptyNode : public XnatNode
{
public:
  explicit XnatEmptyNode();
  virtual ~XnatEmptyNode();

  virtual XnatNode* makeChildNode(int row);
};

#endif
