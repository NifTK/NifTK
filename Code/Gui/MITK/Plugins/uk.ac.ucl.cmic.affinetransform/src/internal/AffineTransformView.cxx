/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>

// Qt
#include <QMessageBox>
#include <QFileDialog>
#include <QTableWidget>
#include <QGridLayout>
#include <QPalette>
#include <QColorGroup>
#include <QDebug>
#include <QInputDialog>

// STL
#include <cmath>
#include <algorithm>
#include <cassert>

// ITK
#include <itkResampleImageFilter.h>
#include <itkVectorResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkVectorLinearInterpolateImageFunction.h>
#include <itkAffineTransform.h>
#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkRGBAPixel.h>
#include <itkRGBPixel.h>
#include <itkTransformFileReader.h>
#include <itkTransformFileWriter.h>
#include <itkPoint.h>
#include <itkBoundingBox.h>

// VTK
#include "vtkLinearTransform.h"
#include "vtkMatrix4x4.h"

// MITK Misc
#include <mitkImage.h>
#include <mitkSurface.h>
#include <mitkImageAccessByItk.h>
#include <mitkITKImageImport.h>
#include <mitkInstantiateAccessFunctions.h>
#include <mitkVector.h> // for PointType;
#include <mitkIDataStorageService.h>
#include <mitkRenderingManager.h>
#include "mitkAffineInteractor3D.h"
#include <mitkEllipsoid.h>
#include <mitkCylinder.h>
#include <mitkCone.h>
#include <mitkCuboid.h>

// NIFTK
#include "AffineTransformView.h"
#include "mitkAffineTransformDataNodeProperty.h"
#include "mitkAffineTransformParametersDataNodeProperty.h"
#include "ConversionUtils.h"

#include <itkImageFileWriter.h>

//-----------------------------------------------------------------------------
AffineTransformView::AffineTransformView()
{
  m_Controls = NULL;
  msp_DataOwnerNode = NULL;
  m_AffineInteractor3D = NULL;
  m_customAxesActor = NULL;
  m_legendActor = NULL;
  m_boundingObject = NULL;
  m_boundingObjectNode = NULL;
  m_currentDataObject = NULL;
  m_inInteractiveMode = false;
  m_legendAdded = false;
  m_rotationMode = false;

  QFile xmlDesc;
  xmlDesc.setFileName(":/AffineTransform/AffineTransformInteractorSM.xml");
  //qDebug() <<xmlDesc.exists();

  if (xmlDesc.exists() && xmlDesc.open(QIODevice::ReadOnly))
  {
    // Make a text stream on the doco  
    QTextStream textStream(&xmlDesc);

    // Read all the contents
    QString qContents = textStream.readAll();
  
    // Load StateMachine patterns
    mitk::GlobalInteraction* globalInteractor =  mitk::GlobalInteraction::GetInstance();
    if (globalInteractor->GetStateMachineFactory()->LoadBehaviorString(qContents.toStdString()))
      qDebug() <<"Loaded the state-machine correctly!";
  }

  m_AffineTransformer = mitk::AffineTransformer::New();
}

//-----------------------------------------------------------------------------
AffineTransformView::~AffineTransformView()
{
  if (m_Controls != NULL)
  {
    delete m_Controls;
  }
  if (m_customAxesActor != NULL)
  {
    delete m_customAxesActor;
  }
}

//-----------------------------------------------------------------------------
void AffineTransformView::CreateQtPartControl( QWidget *parent )
{
  if (!m_Controls)
  {
    // create GUI widgets from the Qt Designer's .ui file
    m_Controls = new Ui::AffineTransformWidget();
    m_Controls->setupUi(parent);

    m_ParentWidget = parent;

    QPalette palette = m_Controls->transformMatrixLabel->palette();
    QColor colour = palette.color(QPalette::Window);
    QString styleSheet = "background-color:rgb(";
    styleSheet.append(QString::number(colour.red()));
    styleSheet.append(",");
    styleSheet.append(QString::number(colour.green()));
    styleSheet.append(",");
    styleSheet.append(QString::number(colour.blue()));
    styleSheet.append(")");
    m_Controls->affineTransformDisplay->setStyleSheet(styleSheet);

    connect(m_Controls->resetButton, SIGNAL(clicked()), this, SLOT(OnResetTransformPushed()));
    connect(m_Controls->resampleButton, SIGNAL(clicked()), this, SLOT(OnResampleTransformPushed()));
    connect(m_Controls->loadButton, SIGNAL(clicked()), this, SLOT(OnLoadTransformPushed()));
    connect(m_Controls->saveButton, SIGNAL(clicked()), this, SLOT(OnSaveTransformPushed()));

    connect(m_Controls->rotationSpinBoxX, SIGNAL(valueChanged(double)), this, SLOT(OnParameterChanged(double)));
    connect(m_Controls->rotationSpinBoxY, SIGNAL(valueChanged(double)), this, SLOT(OnParameterChanged(double)));
    connect(m_Controls->rotationSpinBoxZ, SIGNAL(valueChanged(double)), this, SLOT(OnParameterChanged(double)));
    connect(m_Controls->shearSpinBoxXY, SIGNAL(valueChanged(double)), this, SLOT(OnParameterChanged(double)));
    connect(m_Controls->shearSpinBoxXZ, SIGNAL(valueChanged(double)), this, SLOT(OnParameterChanged(double)));
    connect(m_Controls->shearSpinBoxYZ, SIGNAL(valueChanged(double)), this, SLOT(OnParameterChanged(double)));
    connect(m_Controls->translationSpinBoxX, SIGNAL(valueChanged(double)), this, SLOT(OnParameterChanged(double)));
    connect(m_Controls->translationSpinBoxY, SIGNAL(valueChanged(double)), this, SLOT(OnParameterChanged(double)));
    connect(m_Controls->translationSpinBoxZ, SIGNAL(valueChanged(double)), this, SLOT(OnParameterChanged(double)));
    connect(m_Controls->scalingSpinBoxX, SIGNAL(valueChanged(double)), this, SLOT(OnParameterChanged(double)));
    connect(m_Controls->scalingSpinBoxY, SIGNAL(valueChanged(double)), this, SLOT(OnParameterChanged(double)));
    connect(m_Controls->scalingSpinBoxZ, SIGNAL(valueChanged(double)), this, SLOT(OnParameterChanged(double)));
    connect(m_Controls->centreRotationRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnParameterChanged(bool)));

    connect(m_Controls->checkBox_Interactive, SIGNAL(toggled(bool )), this, SLOT(OnInteractiveModeToggled(bool )));
    connect(m_Controls->radioButton_translate, SIGNAL(toggled(bool )), this, SLOT(OnRotationToggled(bool )));
    connect(m_Controls->radioButton_rotate, SIGNAL(toggled(bool )), this, SLOT(OnRotationToggled(bool )));
    connect(m_Controls->checkBox_fixAngle, SIGNAL(toggled(bool )), this, SLOT(OnFixAngleToggled(bool )));
    connect(m_Controls->radioButton_001, SIGNAL(toggled(bool )), this, SLOT(OnAxisChanged(bool )));
    connect(m_Controls->radioButton_010, SIGNAL(toggled(bool )), this, SLOT(OnAxisChanged(bool )));
    connect(m_Controls->radioButton_100, SIGNAL(toggled(bool )), this, SLOT(OnAxisChanged(bool )));
  }
}

//-----------------------------------------------------------------------------
void AffineTransformView::SetFocus()
{
}

//-----------------------------------------------------------------------------
void AffineTransformView::SetControlsEnabled(bool isEnabled)
{
  m_Controls->rotationSpinBoxX->setEnabled(isEnabled);
  m_Controls->rotationSpinBoxY->setEnabled(isEnabled);
  m_Controls->rotationSpinBoxZ->setEnabled(isEnabled);
  m_Controls->shearSpinBoxXY->setEnabled(isEnabled);
  m_Controls->shearSpinBoxXZ->setEnabled(isEnabled);
  m_Controls->shearSpinBoxYZ->setEnabled(isEnabled);
  m_Controls->translationSpinBoxX->setEnabled(isEnabled);
  m_Controls->translationSpinBoxY->setEnabled(isEnabled);
  m_Controls->translationSpinBoxZ->setEnabled(isEnabled);
  m_Controls->scalingSpinBoxX->setEnabled(isEnabled);
  m_Controls->scalingSpinBoxY->setEnabled(isEnabled);
  m_Controls->scalingSpinBoxZ->setEnabled(isEnabled);
  m_Controls->centreRotationRadioButton->setEnabled(isEnabled);
  m_Controls->resampleButton->setEnabled(isEnabled);
  m_Controls->saveButton->setEnabled(isEnabled);
  m_Controls->loadButton->setEnabled(isEnabled);
  m_Controls->resetButton->setEnabled(isEnabled);
  m_Controls->affineTransformDisplay->setEnabled(isEnabled);
}

//-----------------------------------------------------------------------------
void AffineTransformView::OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes)
{
  if (nodes.size() != 1)
  {
    this->SetControlsEnabled(false);
    return;
  }

  if (nodes[0].IsNull())
  {
    this->SetControlsEnabled(false);
    return;
  }

  if (m_AffineTransformer.IsNull())
    return;

  // Set the current node pointer into the transformer class
  m_AffineTransformer->OnNodeChanged(nodes[0]);

  // Update the controls on the UI based on the datanode's properties
  SetValuesOnUI(m_AffineTransformer->GetCurrentTransformParameters());

  // Update the matrix on the UI
  UpdateTransformDisplay();
  this->SetControlsEnabled(true);

  // Final check, only enable resample button, if current selection is an image.
  mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(msp_DataOwnerNode->GetData());
  if (image.IsNotNull())
  {
    m_Controls->resampleButton->setEnabled(true);
  }
  else
  {
    m_Controls->resampleButton->setEnabled(false);
  }
}

//-----------------------------------------------------------------------------
void AffineTransformView::SetValuesOnUI(mitk::AffineTransformParametersDataNodeProperty::Pointer parametersProperty)
{
  mitk::AffineTransformParametersDataNodeProperty::ParametersType params = parametersProperty->GetAffineTransformParameters();

  m_Controls->rotationSpinBoxX->setValue(params[0]);
  m_Controls->rotationSpinBoxY->setValue(params[1]);
  m_Controls->rotationSpinBoxZ->setValue(params[2]);

  m_Controls->translationSpinBoxX->setValue(params[3]);
  m_Controls->translationSpinBoxY->setValue(params[4]);
  m_Controls->translationSpinBoxZ->setValue(params[5]);

  m_Controls->scalingSpinBoxX->setValue(params[6]);
  m_Controls->scalingSpinBoxY->setValue(params[7]);
  m_Controls->scalingSpinBoxZ->setValue(params[8]);

  m_Controls->shearSpinBoxXY->setValue(params[9]);
  m_Controls->shearSpinBoxXZ->setValue(params[10]);
  m_Controls->shearSpinBoxYZ->setValue(params[11]);

  if (params[12] != 0)
  {
    m_Controls->centreRotationRadioButton->setChecked(true);
  }
  else
  {
    m_Controls->centreRotationRadioButton->setChecked(false);
  }
}

//-----------------------------------------------------------------------------
void AffineTransformView::GetValuesFromUI(mitk::AffineTransformParametersDataNodeProperty::Pointer parametersProperty)
{
  mitk::AffineTransformParametersDataNodeProperty::ParametersType params = parametersProperty->GetAffineTransformParameters();

  params[0] = m_Controls->rotationSpinBoxX->value();
  params[1] = m_Controls->rotationSpinBoxY->value();
  params[2] = m_Controls->rotationSpinBoxZ->value();

  params[3] = m_Controls->translationSpinBoxX->value();
  params[4] = m_Controls->translationSpinBoxY->value();
  params[5] = m_Controls->translationSpinBoxZ->value();

  params[6] = m_Controls->scalingSpinBoxX->value();
  params[7] = m_Controls->scalingSpinBoxY->value();
  params[8] = m_Controls->scalingSpinBoxZ->value();

  params[9] = m_Controls->shearSpinBoxXY->value();
  params[10] = m_Controls->shearSpinBoxXZ->value();
  params[11] = m_Controls->shearSpinBoxYZ->value();

  if (m_Controls->centreRotationRadioButton->isChecked())
  {
    params[12] = 1;
  }
  else
  {
    params[12] = 0;
  }

  parametersProperty->SetAffineTransformParameters(params);
}

//-----------------------------------------------------------------------------
void AffineTransformView::UpdateTransformDisplay() 
{
  // This method gets a 4x4 matrix corresponding to the current transform, given by the current values
  // in all the rotation, translation, scaling and shearing widgets, and outputs the matrix in the GUI.
  // It does not actually change, or recompute, or transform anything. So we are just saying
  // "update the displayed view of the transformation".
  vtkSmartPointer<vtkMatrix4x4> sp_Transform = m_AffineTransformer->GetCurrentTransformMatrix();

  for (int rInd = 0; rInd < 4; rInd++) for (int cInd = 0; cInd < 4; cInd++)
    m_Controls->affineTransformDisplay->setItem(rInd, cInd, new QTableWidgetItem(QString::number(sp_Transform->Element[rInd][cInd])));
}

//-----------------------------------------------------------------------------
void AffineTransformView::OnParameterChanged() 
{
  // Collect the parameters from the UI then send them to the AffineTransformer
  mitk::AffineTransformParametersDataNodeProperty::Pointer parametersProperty = mitk::AffineTransformParametersDataNodeProperty::New();
  GetValuesFromUI(parametersProperty);

  // Pass the parameters to the transformer
  m_AffineTransformer->OnParametersChanged(parametersProperty);

  // Update the views
  QmitkAbstractView::RequestRenderWindowUpdate();

  // Update the matrix on the UI
  UpdateTransformDisplay();
}

//-----------------------------------------------------------------------------
void AffineTransformView::OnResetTransformPushed() 
{
  // Reset the transformation parameters
  ResetTransformation();

  // Update the views
  QmitkAbstractView::RequestRenderWindowUpdate();

  // Update the matrix on the UI
  UpdateTransformDisplay();
}

//-----------------------------------------------------------------------------
void AffineTransformView::ResetTransformation()
{
  // Reset transformation parameters to identity
  mitk::AffineTransformParametersDataNodeProperty::Pointer affineTransformParametersProperty = mitk::AffineTransformParametersDataNodeProperty::New();
  affineTransformParametersProperty->Identity();

  // Update the UI
  SetValuesOnUI(affineTransformParametersProperty);

  // Update the transformer
  m_AffineTransformer->OnParametersChanged(affineTransformParametersProperty);
}

//-----------------------------------------------------------------------------
void AffineTransformView::OnLoadTransformPushed()
{
  QString fileName;
  fileName = QFileDialog::getOpenFileName(NULL, tr("Select transform file"), QString(), tr("ITK affine transform file (*.txt *.tfm);;Any file (*)"));
  if (fileName.length() > 0) 
  {
    m_AffineTransformer->OnLoadTransform(fileName.toStdString());
  }
  OnResetTransformPushed();
}

//-----------------------------------------------------------------------------
void AffineTransformView::OnSaveTransformPushed() 
{

  assert(msp_DataOwnerNode);

  QString fileName;
  fileName = QFileDialog::getSaveFileName(NULL, tr("Destination for transform"), QString(), tr("ITK affine transform file (*.tfm *.txt);;Any file (*)"));
  if (fileName.length() > 0) 
  {
    m_AffineTransformer->OnSaveTransform(fileName.toStdString());
  }
}

//-----------------------------------------------------------------------------
void AffineTransformView::OnResampleTransformPushed() 
{
  // Resample the datanode
  m_AffineTransformer->OnResampleTransform();

  // Then reset the parameters.
  ResetTransformation();
  UpdateTransformDisplay();
  
  // And update the views
  QmitkAbstractView::RequestRenderWindowUpdate();
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

void AffineTransformView::CreateNewBoundingObject(mitk::DataNode::Pointer node)
{
  // attach the cuboid to the image and update the views
  if (node.IsNotNull())
  {
    //m_currentImage = dynamic_cast<mitk::Image*>(node->GetData());
    m_currentDataObject = dynamic_cast<mitk::BaseData*>(node->GetData());

    node->SetBoolProperty( "pickable", true); 

    if(m_currentDataObject.IsNotNull())
    {
      if (this->GetDataStorage()->GetNamedDerivedNode("BoundingObject", node))
      {
        m_boundingObject->FitGeometry(m_currentDataObject->GetTimeSlicedGeometry());
        mitk::RenderingManager::GetInstance()->RequestUpdateAll();
        return;
      }

      bool fitBoundingObject = false;
      
      if(m_boundingObject.IsNull())
      {
          QStringList items;
          items << tr("Cuboid") << tr("Ellipsoid") << tr("Cylinder") << tr("Cone");

          bool ok;
          QString item = QInputDialog::getItem(m_ParentWidget, tr("Select Bounding Object"), tr("Type of Bounding Object:"), items, 0, false, &ok);

          if (!ok)
            return;

          if (item == "Ellipsoid")
            m_boundingObject = mitk::Ellipsoid::New();
          else if(item == "Cylinder")
            m_boundingObject = mitk::Cylinder::New();
          else if (item == "Cone")
            m_boundingObject = mitk::Cone::New();
          else if (item == "Cuboid")
            m_boundingObject = mitk::Cuboid::New();
          else
            return;

          m_boundingObjectNode = mitk::DataNode::New();
          m_boundingObjectNode->SetData(m_boundingObject);
          m_boundingObjectNode->SetProperty( "name", mitk::StringProperty::New( "BoundingObject" ) );
          m_boundingObjectNode->SetProperty( "color", mitk::ColorProperty::New(1.0, 1.0, 0.0) );
          m_boundingObjectNode->SetProperty( "opacity", mitk::FloatProperty::New(0.4) );
          m_boundingObjectNode->SetProperty( "layer", mitk::IntProperty::New(99) ); // arbitrary, copied from segmentation functionality
          m_boundingObjectNode->SetProperty( "helper object", mitk::BoolProperty::New(true) );

          //m_AffineInteractor3D = AffineTransformInteractor3D::New("AffineTransformInteractor", m_boundingObjectNode);
          m_AffineInteractor3D = AffineTransformInteractor3D::New("AffineTransformInteractor", node);
          connect(m_AffineInteractor3D, SIGNAL(transformReady()), this, SLOT(OnTransformReady()) );
          m_AffineInteractor3D->SetBoundingObjectNode(m_boundingObjectNode);

          m_AffineInteractor3D->SetPrecision(3);
          
          if (m_rotationMode)
            m_AffineInteractor3D->SetInteractionModeToRotation();
          else
            m_AffineInteractor3D->SetInteractionModeToTranslation();
          
          fitBoundingObject = true;
      }

      if (m_boundingObject.IsNull())
        return;

      AddBoundingObjectToNode(node, fitBoundingObject);

      node->SetVisibility(true);
      mitk::RenderingManager::GetInstance()->InitializeViews();
      mitk::RenderingManager::GetInstance()->RequestUpdateAll();
      //m_Controls->m_NewBoxButton->setText("Reset bounding box!");
      //m_Controls->btnCrop->setEnabled(true);
    }
  }
  else 
    QMessageBox::information(NULL, "Image cropping functionality", "Load an image first!");
}

void AffineTransformView::AddBoundingObjectToNode(mitk::DataNode::Pointer node, bool fit)
{
  //m_currentImage = dynamic_cast<mitk::Image*>(node->GetData());
  m_currentDataObject = dynamic_cast<mitk::BaseData*>(node->GetData());
  m_AffineInteractor3D->SetInteractionModeToTranslation();

  if(!this->GetDataStorage()->Exists(m_boundingObjectNode))
  {
    this->GetDataStorage()->Add(m_boundingObjectNode, node);
    if (fit)
    {
      m_boundingObject->FitGeometry(m_currentDataObject->GetTimeSlicedGeometry());
    }
    mitk::GlobalInteraction::GetInstance()->AddInteractor( m_AffineInteractor3D );
  }
  m_boundingObjectNode->SetVisibility(true);
}

void AffineTransformView::RemoveBoundingObjectFromNode()
{
  if (m_boundingObjectNode.IsNotNull())
  {
    if(this->GetDataStorage()->Exists(m_boundingObjectNode))
    {
      this->GetDataStorage()->Remove(m_boundingObjectNode);
      mitk::GlobalInteraction::GetInstance()->RemoveInteractor(m_AffineInteractor3D);
    }
  }
}

void AffineTransformView::OnInteractiveModeToggled(bool on)
{
  if (on)
  {
    m_inInteractiveMode = true;

    //this->GetDataManagerSelection().at(0)->IsVisible
    if (msp_DataOwnerNode->IsVisible(0))
      this->CreateNewBoundingObject(msp_DataOwnerNode);

    this->DisplayLegends(true);
    mitk::RenderingManager::GetInstance()->RequestUpdateAll();
  }
  else
     m_inInteractiveMode = false;

    //if (msp_DataOwnerNode->IsVisible(0) && m_inInteractiveMode)
    //this->CreateNewBoundingObject(msp_DataOwnerNode);
}

void AffineTransformView::OnRotationToggled(bool on)
{
  if (!m_Controls->radioButton_translate->isChecked() && m_Controls->radioButton_rotate->isChecked())
    m_rotationMode = true;
  else if (m_Controls->radioButton_translate->isChecked() && !m_Controls->radioButton_rotate->isChecked())
    m_rotationMode = false;

  if (m_AffineInteractor3D.IsNotNull())
  {
    if (m_rotationMode == false)
      m_AffineInteractor3D->SetInteractionModeToTranslation();
    else
      m_AffineInteractor3D->SetInteractionModeToRotation();
  }
}

void AffineTransformView::OnFixAngleToggled(bool on)
{
  if (on == true)
  {
    m_Controls->radioButton_001->setEnabled(true);
    m_Controls->radioButton_010->setEnabled(true);
    m_Controls->radioButton_100->setEnabled(true);

    if (m_AffineInteractor3D.IsNotNull())
    {
      if (m_Controls->radioButton_001->isChecked())
        m_AffineInteractor3D->SetAxesFixed(true, 0);
      else if (m_Controls->radioButton_010->isChecked())
        m_AffineInteractor3D->SetAxesFixed(true, 1);
      else if (m_Controls->radioButton_100->isChecked())
        m_AffineInteractor3D->SetAxesFixed(true, 2);
    }
  }
  else
  {
    m_Controls->radioButton_001->setEnabled(false);
    m_Controls->radioButton_010->setEnabled(false);
    m_Controls->radioButton_100->setEnabled(false);

    if (m_AffineInteractor3D.IsNotNull())
      m_AffineInteractor3D->SetAxesFixed(false);
  }
}

void AffineTransformView::OnAxisChanged(bool on)
{
  if (m_AffineInteractor3D.IsNotNull())
  {
    if (m_Controls->radioButton_001->isChecked())
      m_AffineInteractor3D->SetAxesFixed(true, 0);
    else if (m_Controls->radioButton_010->isChecked())
      m_AffineInteractor3D->SetAxesFixed(true, 1);
    else if (m_Controls->radioButton_100->isChecked())
      m_AffineInteractor3D->SetAxesFixed(true, 2);
  }
}


void AffineTransformView::OnTransformReady()
{
  mitk::Geometry3D::Pointer geom = msp_DataOwnerNode->GetData()->GetGeometry();
  vtkMatrix4x4 * currentMat = geom->GetVtkTransform()->GetMatrix();

  for (int rInd = 0; rInd < 4; rInd++) for (int cInd = 0; cInd < 4; cInd++)
		m_Controls->affineTransformDisplay->setItem(rInd, cInd, new QTableWidgetItem(QString::number(currentMat->Element[rInd][cInd])));

}

bool AffineTransformView::DisplayLegends(bool legendsON)
{
  //mitk::BaseRenderer::GetInstance( mitk::BaseRenderer::GetRenderWindowByName("stdmulti.widget1")))
  QmitkRenderWindow * qRenderWindow = this->GetRenderWindowPart()->GetQmitkRenderWindow("3d");
  vtkRenderWindow *renderWindow = NULL;
  vtkRenderWindowInteractor *renderWindowInteractor = NULL;
  vtkRenderer *currentVtkRenderer = NULL;
  //vtkCamera *camera = NULL;

  if (qRenderWindow != NULL )
  {
    renderWindow = qRenderWindow->GetVtkRenderWindow();
    if ( renderWindow != NULL )
    {
      renderWindowInteractor = renderWindow->GetInteractor();
      if ( renderWindowInteractor != NULL )
      {
        currentVtkRenderer = renderWindowInteractor->GetInteractorStyle()->GetCurrentRenderer();
        
        if (currentVtkRenderer == NULL)
          return false;
      }
      else return false;
    }
    else return false;
  }
  else return false;

  if (currentVtkRenderer != NULL)
  {
    if (legendsON)
    {
      if (!m_legendAdded)
      {
        m_legendActor = vtkLegendScaleActor::New();
        currentVtkRenderer->AddActor(m_legendActor);

        m_axesActor = vtkAxesActor::New();
        m_axesActor->SetShaftTypeToCylinder();
        m_axesActor->SetXAxisLabelText( "X" );
        m_axesActor->SetYAxisLabelText( "Y" );
        m_axesActor->SetZAxisLabelText( "Z" );
        m_axesActor->SetTotalLength(200, 200, 200);
        m_axesActor->AxisLabelsOn();
        m_axesActor->SetCylinderRadius(0.02);
        m_axesActor->SetConeRadius(0.2);
        m_axesActor->SetPosition(0.0, 0.0, 0.0);
        m_axesActor->SetOrigin(0.0, 0.0, 0.0);

        m_customAxesActor = new CustomVTKAxesActor();
        m_customAxesActor->SetShaftTypeToCylinder();
        m_customAxesActor->SetXAxisLabelText("X");
        m_customAxesActor->SetYAxisLabelText("Y");
        m_customAxesActor->SetZAxisLabelText("Z");
        m_customAxesActor->SetTotalLength(150, 150, 150);
        m_customAxesActor->AxisLabelsOn();
        m_customAxesActor->SetCylinderRadius(0.02);
        m_customAxesActor->SetConeRadius(0.2);
        m_customAxesActor->SetPosition(0.0, 0.0, 0.0);
        m_customAxesActor->SetOrigin(0.0, 0.0, 0.0);
        
        //m_axesActor->SetShaftTypeToCylinder();
        

        //currentVtkRenderer->AddActor(m_axesActor);
        currentVtkRenderer->AddActor(m_customAxesActor);

        m_legendAdded = true;

      }
      //nothing to do if already added
    }
    else
    {
      if (m_legendActor != NULL)
      {
        currentVtkRenderer->RemoveActor(m_legendActor);
        m_legendActor->Delete();
        m_legendActor = NULL;
      }
      if (m_axesActor != NULL)
      {
        currentVtkRenderer->RemoveActor(m_axesActor);
        m_axesActor->Delete();
        m_axesActor = NULL;
      }
      m_legendAdded = false;
    }
  }
  else return false;

  return true;
}
