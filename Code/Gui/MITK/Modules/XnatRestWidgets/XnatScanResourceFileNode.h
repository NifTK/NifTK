#ifndef XnatScanResourceFileNode_h
#define XnatScanResourceFileNode_h

#include "XnatRestWidgetsExports.h"

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatScanResourceFileNode : public XnatNode
{
public:
  explicit XnatScanResourceFileNode(int row = -1, XnatNode* parent = NULL);
  virtual ~XnatScanResourceFileNode();

  virtual XnatNode* makeChildNode(int row);
  virtual void download(int row, const char* zipFilename);

  virtual bool isFile() const;
};

#endif
