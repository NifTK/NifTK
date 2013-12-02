/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkRMSErrorWidget_h
#define QmitkRMSErrorWidget_h

#include "niftkIGIGuiExports.h"
#include "ui_QmitkRMSErrorWidget.h"
#include <QWidget>
#include <mitkDataStorage.h>
#include <vtkMatrix4x4.h>

/**
 * \class QmitkRMSErrorWidget
 * \brief Simple helper widget to allow user to select targets and moving points, and
 * every time you set a transformation matrix, will update the text edit field to 
 * display the RMS error between the transformed moving points and the fixed points.
 */
class NIFTKIGIGUI_EXPORT QmitkRMSErrorWidget : public QWidget, public Ui_QmitkRMSErrorWidget
{
  Q_OBJECT

public:

  QmitkRMSErrorWidget(QWidget *parent = 0);
  virtual ~QmitkRMSErrorWidget();

  void SetDataStorage(const mitk::DataStorage* dataStorage);
  double UpdateTransformation(const vtkMatrix4x4& matrix);
  
private slots:

private:
  mitk::DataStorage::Pointer m_DataStorage;
};

#endif // QmitkRMSErrorWidget_h
