#ifndef XnatCategoryNode_h
#define XnatCategoryNode_h

#include "XnatRestWidgetsExports.h"

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatCategoryNode : public XnatNode
{
public:
  explicit XnatCategoryNode(int row = -1, XnatNode* parent = NULL);
  virtual ~XnatCategoryNode();

  virtual XnatNode* makeChildNode(int row);
  virtual void download(int row, const char* zipFilename);
  virtual void add(int row, const char* reconstruction);

  virtual const char* getModifiableChildKind(int row) const;
  virtual const char* getModifiableParentName(int row) const;

  virtual bool holdsFiles() const;
  virtual bool isModifiable(int row) const;

};

#endif
