/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVLVideoOverlayWidget_h
#define niftkVLVideoOverlayWidget_h

// Note:
// The None constant is defined by Xlib.h and it is also declared as an enum
// in qstyleoption.h. This causes a compile error with gcc. As a workaround,
// we undefine the symbol before including the Qt headers. This works only
// if cotire is switched off.
#ifdef None
#undef None
#endif

#include "ui_niftkVLVideoOverlayWidget.h"
#include "niftkVLExports.h"
#include <QWidget>
#include <mitkDataStorage.h>

class VLQtWidget;
class OclResourceService;
class QHBoxLayout;

namespace niftk
{

/**
* \class QmitkIGIVLEditor
* \see IGIVLEditor
*/
class NIFTKVL_EXPORT VLVideoOverlayWidget : public QWidget, public Ui_VLVideoOverlayWidget
{

  Q_OBJECT

public:

  VLVideoOverlayWidget(QWidget *parent);
  virtual ~VLVideoOverlayWidget();

  void SetOclResourceService(OclResourceService* oclserv);
  void SetBackgroundColour(unsigned int aabbggrr);
  void SetEyeHandFileName(const std::string& fileName) {} // ToDo
  void SetDataStorage(mitk::DataStorage* storage);

private slots:

  void OnLeftOverlayCheckBoxChecked(bool);
  void OnRightOverlayCheckBoxChecked(bool);
  void On3DViewerCheckBoxChecked(bool);
  void OnTrackedViewerCheckBoxChecked(bool);
  void OnOpacitySliderMoved(int);
  void OnLeftImageSelected(const mitk::DataNode* node);
  void OnRightImageSelected(const mitk::DataNode* node);
  void OnTransformSelected(const mitk::DataNode* node);

private:

  VLVideoOverlayWidget(const VLVideoOverlayWidget&);  // Purposefully not implemented.
  void operator=(const VLVideoOverlayWidget&);  // Purposefully not implemented.

  /**
   * \brief Utility method to deregister data storage listeners.
   */
  void DeRegisterDataStorageListeners();

  /**
   * \brief Called when a DataStorage Node Changed Event was emitted.
   */
  void NodeChanged(const mitk::DataNode* node);

  mitk::DataStorage::Pointer m_DataStorage;

  QHBoxLayout* m_HorizontalLayout;
  QWidget*     m_OverlayViewers;
  VLQtWidget*  m_LeftOverlayViewer;
  VLQtWidget*  m_RightOverlayViewer;
  VLQtWidget*  m_TrackedViewer;
  VLQtWidget*  m_3DViewer;
};

} // end namespace

#endif
