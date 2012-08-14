#ifndef XNATNODE_H
#define XNATNODE_H

#include <QString>
#include <QList>

class XnatNodeActivity;

class XnatNode
{
public:
  XnatNode(XnatNodeActivity& activity, int row = -1, XnatNode* parent = NULL);
  ~XnatNode();

  const char* getParentName();
  int getRowInParent();
  XnatNode* getParentNode();
  int getNumChildren();
  const char* getChildName(int row);
  XnatNode* getChildNode(int row);

  void addChild(const char* name);
  void addChildNode(int row, XnatNode* node);
  XnatNode* makeChildNode(int row);
  void removeChildNode(int row);

  void download(int row, const char* zipFilename);
  void downloadAllFiles(int row, const char* zipFilename);
  void upload(int row, const char* zipFilename);
  void add(int row, const char* name);
  void remove(int row);

  const char* getKind();
  const char* getModifiableChildKind(int row);
  const char* getModifiableParentName(int row);

  bool isFile();
  bool holdsFiles();
  bool receivesFiles();
  bool isModifiable(int row);
  bool isDeletable();

private:
  class XnatChild
  {
  public:
    QString name;
    XnatNode* node;

    XnatChild(const char* name, XnatNode* node = 0);
    ~XnatChild();
  };

  XnatNodeActivity& nodeActivity;
  int rowInParent;
  XnatNode* parent;
  QList<XnatChild*> children;
};

#endif
