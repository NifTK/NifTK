#ifndef XnatScanResourceNode_h
#define XnatScanResourceNode_h

#include "XnatRestWidgetsExports.h"

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatScanResourceNode : public XnatNode
{
public:
  explicit XnatScanResourceNode(int row = -1, XnatNode* parent = NULL);
  virtual ~XnatScanResourceNode();

  virtual XnatNode* makeChildNode(int row);
  virtual void download(int row, const char* zipFilename);

  virtual const char* getKind() const;
  virtual bool holdsFiles() const;
};

#endif
