#ifndef XnatNode_h
#define XnatNode_h

#include <string>

#include <QList>
#include <vector>

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
    std::string name;
    XnatNode* node;

    XnatChild(const char* name, XnatNode* node = NULL);
    ~XnatChild();
  };

  XnatNodeActivity& nodeActivity;
  int rowInParent;
  XnatNode* parent;
  std::vector<XnatChild*> children;
};

#endif
