#ifndef XnatSubjectNode_h
#define XnatSubjectNode_h

#include "XnatRestWidgetsExports.h"

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatSubjectNode : public XnatNode
{
public:
  explicit XnatSubjectNode(int row = -1, XnatNode* parent = NULL);
  virtual ~XnatSubjectNode();

  virtual XnatNode* makeChildNode(int row);

  virtual const char* getKind() const;
};

#endif
