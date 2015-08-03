/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef PivotCalibrationView_h
#define PivotCalibrationView_h

#include <QmitkBaseView.h>
#include "ui_PivotCalibrationView.h"

#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

#include <service/event/ctkEvent.h>


/**
 * \class PivotCalibrationView
 * \brief User interface to provide controls for Ultrasound CT registration.
 *
 * \ingroup uk_ac_ucl_cmic_igiultrasoundctreg_internal
*/
class PivotCalibrationView : public QmitkBaseView
{  
  /**
   * this is needed for all Qt objects that should have a Qt meta-object
   * (everything that derives from QObject and wants to have signal/slots)
   */
  Q_OBJECT

public:

  PivotCalibrationView();
  virtual ~PivotCalibrationView();

  /**
   * \brief Static view ID = uk.ac.ucl.cmic.igiultrasoundctreg
   */
  static const char* VIEW_ID;

  /**
   * \brief Returns the view ID.
   */

  virtual std::string GetViewID() const;  


protected:

  /**
   *  \brief Called by framework, this method creates all the controls for this view
   */
  virtual void CreateQtPartControl(QWidget *parent);

  /**
   * \brief Called by framework, sets the focus on a specific widget.
   */
  virtual void SetFocus(); 

signals:


protected slots:
  

protected:
   

private slots:

  void DataStorageEventListener(const mitk::DataNode* node);  

  void OnPivotCalibrationButtonClicked();
  void OnSaveToFileButtonClicked();

private:


  /**
   * \brief All the controls for the main view part.
   */
  Ui::PivotCalibrationView *m_Controls;
  vtkSmartPointer<vtkMatrix4x4> m_Matrix;

  std::string m_OutputDirectory;
};

#endif // PivotCalibrationView_h
