/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkMatrixWidget_h
#define QmitkMatrixWidget_h

#include "niftkIGIGuiExports.h"
#include "ui_QmitkMatrixWidget.h"
#include <QWidget>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

/**
 * \class QmitkMatrixWidget
 * \brief Widget to provide a matrix that can be cleared, and re-loaded from file.
 */
class NIFTKIGIGUI_EXPORT QmitkMatrixWidget : public QWidget, public Ui_QmitkMatrixWidget
{
  Q_OBJECT

public:

  QmitkMatrixWidget(QWidget *parent = 0);
  virtual ~QmitkMatrixWidget();

  void SynchroniseWidgetWithMatrix();

  void SetClearButtonVisible(const bool& isVisible);
  void SetLoadButtonVisible(const bool& isVisible);

  void SetMatrix(const vtkMatrix4x4& matrix);

  /**
   * \brief Copies the internal matrix, returning a new matrix that the caller is responsible for deleting.
   */
  vtkMatrix4x4* CloneMatrix() const;

signals:
  void MatrixChanged();

private slots:
  void OnClearButtonPressed();
  void OnLoadButtonPressed();

private:
  vtkSmartPointer<vtkMatrix4x4> m_Matrix;
};

#endif // QmitkMatrixWidget_h
