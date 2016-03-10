/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGIVLEditor_h
#define QmitkIGIVLEditor_h

// Note:
// The None constant is defined by Xlib.h and it is also declared as an enum
// in qstyleoption.h. This causes a compile error with gcc. As a workaround,
// we undefine the symbol before including the Qt headers. This works only
// if cotire is switched off.
#ifdef None
#undef None
#endif

#include "ui_QmitkIGIVLEditor.h"
#include "niftkVLExports.h"
#include <QWidget>
#include <mitkColorProperty.h>
#include <mitkDataStorage.h>


class QmitkRenderWindow;
class VLQt4Widget;
class OclResourceService;


/**
 * \class QmitkIGIVLEditor
 * \see IGIVLEditor
 */
class NIFTKVL_EXPORT QmitkIGIVLEditor : public QWidget, public Ui_QmitkIGIVLEditor
{

  Q_OBJECT

public:

  QmitkIGIVLEditor(QWidget *parent);
  virtual ~QmitkIGIVLEditor();

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

  QmitkIGIVLEditor(const QmitkIGIVLEditor&);  // Purposefully not implemented.
  void operator=(const QmitkIGIVLEditor&);  // Purposefully not implemented.

  /**
   * \brief Utility method to deregister data storage listeners.
   */
  void DeRegisterDataStorageListeners();
  
  /**
   * \brief Called when a DataStorage Node Changed Event was emitted.
   */
  void NodeChanged(const mitk::DataNode* node);
  
  mitk::DataStorage::Pointer m_DataStorage;

  VLQt4Widget*    m_OverlayViewer;
  VLQt4Widget*    m_3DViewer;
};

#endif // QmitkIGIVLEditor_h
