/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkLaparoscopicSurgeryManager_h
#define QmitkLaparoscopicSurgeryManager_h

#include "niftkIGIGuiExports.h"
#include "ui_QmitkLaparoscopicSurgeryManager.h"
#include <itkObject.h>
#include <mitkDataStorage.h>
#include <QWidget>

/**
 * \class QmitkLaparoscopicSurgeryManager
 * \brief Class to manage the Laparoscopic Surgery Functionality (Liver project).
 */
class NIFTKIGIGUI_EXPORT QmitkLaparoscopicSurgeryManager : public QWidget, public Ui_QmitkLaparoscopicSurgeryManager, public itk::Object
{

  Q_OBJECT

public:

  mitkClassMacro(QmitkLaparoscopicSurgeryManager, itk::Object);
  itkNewMacro(QmitkLaparoscopicSurgeryManager);

  /**
   * \brief Creates the base class widgets, and connects signals and slots.
   */
  void setupUi(QWidget* parent);

  /**
   * \brief Set the Data Storage
   * \param dataStorage An MITK DataStorage, which is stored internally.
   */
  void SetDataStorage(mitk::DataStorage* dataStorage);

  /**
   * \brief Get the Data Storage that this tool manager is currently connected to.
   */
  itkGetConstMacro(DataStorage, mitk::DataStorage*);

  /**
   * \brief Updates the whole scene.
   */
  void Update();

signals:

protected:

  QmitkLaparoscopicSurgeryManager(); // Purposefully hidden
  virtual ~QmitkLaparoscopicSurgeryManager(); // Purposefully hidden

  QmitkLaparoscopicSurgeryManager(const QmitkLaparoscopicSurgeryManager&); // Purposefully not implemented.
  QmitkLaparoscopicSurgeryManager& operator=(const QmitkLaparoscopicSurgeryManager&); // Purposefully not implemented.

private slots:

private:

  mitk::DataStorage *m_DataStorage;

}; // end class

#endif

