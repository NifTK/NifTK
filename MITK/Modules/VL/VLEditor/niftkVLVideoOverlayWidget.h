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

class OclResourceService;
class QHBoxLayout;

namespace niftk
{

class VLWidget;

/**
* \class VLVideoOverlayWidget
* \brief Contains 4 coordinated VL views for AR purposes (e.g. liver project).
* \see niftk::IGIVLVideoOverlayEditor
*/
class NIFTKVL_EXPORT VLVideoOverlayWidget : public QWidget, public Ui_VLVideoOverlayWidget
{

  Q_OBJECT

public:

  VLVideoOverlayWidget(QWidget *parent);
  virtual ~VLVideoOverlayWidget();

  void SetBackgroundColour(unsigned int aabbggrr);
  void SetEyeHandFileName(const std::string& fileName);
  void SetDataStorage(mitk::DataStorage* storage);
  void SetControlWidgetsVisible(bool visible);
  void SetLeftViewerVisible(bool visible);
  void SetRightViewerVisible(bool visible);
  void Set3DViewerVisible(bool visible);
  void SetTrackingViewerVisible(bool visible);
  void SetOpacity(int betweenZeroAnd100);
  void SetLeftImage(const mitk::DataNode* node);
  void SetRightImage(const mitk::DataNode* node);
  void SetTransform(const mitk::DataNode* node);

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
  VLWidget*    m_LeftOverlayViewer;
  VLWidget*    m_RightOverlayViewer;
  VLWidget*    m_TrackedViewer;
  VLWidget*    m_3DViewer;
};

} // end namespace

#endif
