#ifndef XnatReconstructionResourceFileNode_h
#define XnatReconstructionResourceFileNode_h

#include "XnatRestWidgetsExports.h"

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatReconstructionResourceFileNode : public XnatNode
{
public:
  explicit XnatReconstructionResourceFileNode(int row = -1, XnatNode* parent = NULL);
  virtual ~XnatReconstructionResourceFileNode();

  virtual XnatNode* makeChildNode(int row);
  virtual void download(int row, const char* zipFilename);
  virtual void remove(int row);

  virtual bool isFile() const;
  virtual bool isDeletable() const;
};

#endif
