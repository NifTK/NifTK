#ifndef XNATNODE_H
#define XNATNODE_H

#include <string>
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


inline XnatNode::XnatNode(XnatNodeActivity& a, int r, XnatNode* p) :
                              nodeActivity(a), rowInParent(r), parent(p) {}


inline const char* XnatNode::getParentName()
    { return ( ( parent != NULL ) ? parent->getChildName(rowInParent) : NULL ); }

inline int XnatNode::getRowInParent() { return rowInParent; }

inline XnatNode* XnatNode::getParentNode() { return parent; }

inline int XnatNode::getNumChildren() { return children.size(); }

inline const char* XnatNode::getChildName(int row) { return children[row]->name.c_str(); }

inline XnatNode* XnatNode::getChildNode(int row) { return children[row]->node; }

inline void XnatNode::addChild(const char* name) { children.push_back(new XnatChild(name)); }

inline void XnatNode::addChildNode(int row, XnatNode* node) { children[row]->node = node; }


inline XnatNode::XnatChild::XnatChild(const char* who, XnatNode* ptr) : name(who), node(ptr) {}


#endif
