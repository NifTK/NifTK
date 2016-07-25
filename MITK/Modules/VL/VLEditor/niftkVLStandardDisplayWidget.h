/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVLStandardDisplayWidget_h
#define niftkVLStandardDisplayWidget_h

// Note:
// The None constant is defined by Xlib.h and it is also declared as an enum
// in qstyleoption.h. This causes a compile error with gcc. As a workaround,
// we undefine the symbol before including the Qt headers. This works only
// if cotire is switched off.
#ifdef None
#undef None
#endif

#include "ui_niftkVLStandardDisplayWidget.h"
#include "niftkVLExports.h"
#include <QWidget>
#include <mitkDataStorage.h>

class VLQtWidget;
class QHBoxLayout;

namespace niftk
{

/**
* \class QmitkIGIVLEditor
* \see IGIVLEditor
*/
class NIFTKVL_EXPORT VLStandardDisplayWidget : public QWidget, public Ui_VLStandardDisplayWidget
{

  Q_OBJECT

public:

  VLStandardDisplayWidget(QWidget *parent);
  virtual ~VLStandardDisplayWidget();

  void SetBackgroundColour(unsigned int aabbggrr);
  void SetDataStorage(mitk::DataStorage* storage);

private slots:

private:

  VLStandardDisplayWidget(const VLStandardDisplayWidget&);  // Purposefully not implemented.
  void operator=(const VLStandardDisplayWidget&);  // Purposefully not implemented.

  /**
   * \brief Utility method to deregister data storage listeners.
   */
  void DeRegisterDataStorageListeners();

  /**
   * \brief Called when a DataStorage Node Changed Event was emitted.
   */
  void NodeChanged(const mitk::DataNode* node);

  mitk::DataStorage::Pointer m_DataStorage;

  VLQtWidget*  m_AxialViewer;
  VLQtWidget*  m_SagittalViewer;
  VLQtWidget*  m_CoronalViewer;
  VLQtWidget*  m_3DViewer;
};

} // end namespace

#endif
