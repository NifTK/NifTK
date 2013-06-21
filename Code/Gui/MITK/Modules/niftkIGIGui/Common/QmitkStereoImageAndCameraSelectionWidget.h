/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkStereoImageAndCameraSelectionWidget_h
#define QmitkStereoImageAndCameraSelectionWidget_h

#include "niftkIGIGuiExports.h"
#include "ui_QmitkStereoImageAndCameraSelectionWidget.h"
#include <QWidget>
#include <mitkImage.h>
#include <mitkCoordinateAxesData.h>
#include <mitkDataStorage.h>

/**
 * \class QmitkStereoImageAndCameraSelectionWidget
 * \brief Used to group widgets to select a left and right image and a camera to world transformation.
 */
class NIFTKIGIGUI_EXPORT QmitkStereoImageAndCameraSelectionWidget : public QWidget, public Ui_QmitkStereoImageAndCameraSelectionWidget
{
  Q_OBJECT

public:

  QmitkStereoImageAndCameraSelectionWidget(QWidget *parent = 0);
  virtual ~QmitkStereoImageAndCameraSelectionWidget();

  mitk::Image* GetLeftImage() const;
  mitk::DataNode* GetLeftNode() const;
  mitk::Image* GetRightImage() const;
  mitk::DataNode* GetRightNode() const;
  mitk::CoordinateAxesData* GetCameraTransform() const;
  mitk::DataNode* GetCameraNode() const;

  void SetDataStorage(const mitk::DataStorage* dataStorage);

  void UpdateNodeNameComboBox();

private slots:
  void OnComboBoxIndexChanged(int index);

private:
  mitk::DataStorage::Pointer m_DataStorage;
};

#endif // QmitkStereoImageAndCameraSelectionWidget_h
