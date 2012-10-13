#ifndef XnatReconstructionResourceNode_h
#define XnatReconstructionResourceNode_h

#include "XnatRestWidgetsExports.h"

#include "XnatNode.h"

class XnatRestWidgets_EXPORT XnatReconstructionResourceNode : public XnatNode
{
public:
  explicit XnatReconstructionResourceNode(int row = -1, XnatNode* parent = NULL);
  virtual ~XnatReconstructionResourceNode();

  virtual XnatNode* makeChildNode(int row);
  virtual void download(int row, const char* zipFilename);
  virtual void upload(int row, const char* zipFilename);
  virtual void remove(int row);

  virtual const char* getKind() const;
  virtual bool holdsFiles() const;
  virtual bool receivesFiles() const;
  virtual bool isDeletable() const;
};

#endif
