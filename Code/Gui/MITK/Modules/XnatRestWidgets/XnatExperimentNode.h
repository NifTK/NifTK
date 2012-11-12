#ifndef XnatExperimentNode_h
#define XnatExperimentNode_h

#include "XnatRestWidgetsExports.h"

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatExperimentNode : public XnatNode
{
public:
  explicit XnatExperimentNode(int row = -1, XnatNode* parent = NULL);
  virtual ~XnatExperimentNode();

  virtual XnatNode* makeChildNode(int row);
  virtual void add(int row, const char* reconstruction);

  virtual const char* getKind() const;
  virtual const char* getModifiableChildKind(int row) const;
  virtual const char* getModifiableParentName(int row) const;

  virtual bool isModifiable(int row) const;
};

#endif
