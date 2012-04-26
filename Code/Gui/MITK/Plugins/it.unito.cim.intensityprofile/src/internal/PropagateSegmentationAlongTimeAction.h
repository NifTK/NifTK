/*=============================================================================

 KMaps:     An image processing toolkit for DCE-MRI analysis developed
            at the Molecular Imaging Center at University of Torino.

 See:       http://www.cim.unito.it

 Author:    Miklos Espak <espakm@gmail.com>

 Copyright (c) Miklos Espak
 All Rights Reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef PROPAGATESEGMENTATIONALONGTIMEACTION_H
#define PROPAGATESEGMENTATIONALONGTIMEACTION_H

#include <mitkIContextMenuAction.h>

#include <QObject>

class PropagateSegmentationAlongTimeAction: public QObject, public mitk::IContextMenuAction
{
  Q_OBJECT
  Q_INTERFACES(mitk::IContextMenuAction)

  public:

    PropagateSegmentationAlongTimeAction();
    virtual ~PropagateSegmentationAlongTimeAction();

    //interface methods
    void Run(const QList<mitk::DataNode::Pointer>& selectedNodes);
    void SetDataStorage(mitk::DataStorage* dataStorage);
    void SetStdMultiWidget(QmitkStdMultiWidget *stdMultiWidget);
    void SetSmoothed(bool smoothed);
    void SetDecimated(bool decimated);
    void SetFunctionality(berry::QtViewPart* functionality);

  protected:

     typedef std::vector<mitk::DataNode*> NodeList;

     mitk::DataStorage::Pointer m_DataStorage;
};

#endif // PROPAGATESEGMENTATIONALONGTIMEACTION_H
