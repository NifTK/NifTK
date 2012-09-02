#ifndef XNATREACTIONSAVEDATA_H 
#define XNATREACTIONSAVEDATA_H 

#include "XnatUploadManager.h"
#include "pqReaction.h"


// reaction to save data files to XNAT
class XnatReactionSaveData : public pqReaction
{
    Q_OBJECT
    typedef pqReaction Superclass;

public:
    // Constructor -- Parent cannot be NULL
    XnatReactionSaveData(QAction* parent, XnatUploadManager* uploadMngr, QWidget* p);

    void saveSelectedData();

    // save data files from active port; uses the vtkSMWriterFactory to decide
    // what writes are available; returns true if the creation is successful, 
    // otherwise returns false
    bool saveActiveData(const QString& files);

public slots:
    // updates the enabled state -- Applications need not explicitly call this
    void updateEnableState();

protected:
    // called when the action is triggered
    virtual void onTriggered() { saveSelectedData(); }

private:
    XnatUploadManager* uploadManager;
    QWidget* parent;

    XnatReactionSaveData(const XnatReactionSaveData&);  // Not implemented
    void operator=(const XnatReactionSaveData&);        // Not implemented
};

#endif
