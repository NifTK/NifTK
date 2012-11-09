#ifndef XnatProjectNode_h
#define XnatProjectNode_h

#include "XnatRestWidgetsExports.h"

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatProjectNode : public XnatNode
{
public:
  explicit XnatProjectNode(int row = -1, XnatNode* parent = NULL);
  virtual ~XnatProjectNode();

  virtual XnatNode* makeChildNode(int row);

  virtual const char* getKind() const;

};

#endif
