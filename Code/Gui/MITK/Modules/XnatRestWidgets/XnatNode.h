#ifndef XnatNode_h
#define XnatNode_h

#include "XnatRestWidgetsExports.h"

#include <string>

#include <QList>
#include <vector>


class XnatRestWidgets_EXPORT XnatNode
{
public:
  explicit XnatNode(int row = -1, XnatNode* parent = NULL);
  virtual ~XnatNode();

  const char* getParentName() const;
  int getRowInParent();
  XnatNode* getParentNode();
  int getNumChildren();
  const char* getChildName(int row) const;
  XnatNode* getChildNode(int row) const;

  void addChild(const char* name);
  void setChildNode(int row, XnatNode* node);
  virtual XnatNode* makeChildNode(int row) = 0;
  void removeChildNode(int row);

  virtual void download(int row, const char* zipFilename);
  virtual void upload(int row, const char* zipFilename);
  virtual void add(int row, const char* name);
  virtual void remove(int row);

  virtual const char* getKind() const;
  virtual const char* getModifiableChildKind(int row) const;
  virtual const char* getModifiableParentName(int row) const;

  virtual bool isFile() const;
  virtual bool holdsFiles() const;
  virtual bool receivesFiles() const;
  virtual bool isModifiable(int row) const;
  virtual bool isDeletable() const;

private:
  class XnatChild
  {
  public:
    std::string name;
    XnatNode* node;

    XnatChild(const char* name);
    ~XnatChild();
  };

private:
  int rowInParent;
  XnatNode* parent;
  std::vector<XnatChild*> children;
};

#endif
