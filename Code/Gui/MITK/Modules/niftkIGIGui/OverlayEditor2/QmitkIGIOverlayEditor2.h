/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGIOverlayEditor2_h
#define QmitkIGIOverlayEditor2_h

#include "ui_QmitkIGIOverlayEditor2.h"
#include "niftkIGIGuiExports.h"
#include <QWidget>
#include <mitkColorProperty.h>
#include <mitkDataStorage.h>


class QmitkRenderWindow;

/**
 * \class QmitkIGIOverlayEditor2
 * \see IGIOverlayEditor2
 */
class NIFTKIGIGUI_EXPORT QmitkIGIOverlayEditor2 : public QWidget, public Ui_QmitkIGIOverlayEditor2
{

  Q_OBJECT

public:

  QmitkIGIOverlayEditor2(QWidget *parent);
  virtual ~QmitkIGIOverlayEditor2();

  void SetOclResourceService(OclResourceService* oclserv);

  void SetBackgroundColour(unsigned int aabbggrr);

  void SetDataStorage(mitk::DataStorage* storage);

  /**
   * \brief Called by framework (event from ctkEventAdmin), to indicate that an update should be performed.
   */
  void Update();

private slots:

  void OnOverlayCheckBoxChecked(bool);
  void On3DViewerCheckBoxChecked(bool);
  void OnOpacitySliderMoved(int);
  void OnImageSelected(const mitk::DataNode* node);
  void OnTransformSelected(const mitk::DataNode* node);

private:

  QmitkIGIOverlayEditor2(const QmitkIGIOverlayEditor2&);  // Purposefully not implemented.
  void operator=(const QmitkIGIOverlayEditor2&);  // Purposefully not implemented.

  /**
   * \brief Utility method to deregister data storage listeners.
   */
  void DeRegisterDataStorageListeners();
  
  /**
   * \brief Called when a DataStorage Node Changed Event was emitted.
   */
  void NodeChanged(const mitk::DataNode* node);
  
  mitk::DataStorage::Pointer m_DataStorage;
};

#endif // QmitkIGIOverlayEditor2_h
