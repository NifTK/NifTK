/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Rev$
 Last modified by  : $Author$

 Original author   : stian.johnsen.09@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 

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

const std::string AffineTransformView::VIEW_ID                   = "uk.ac.ucl.cmic.affinetransformview";
const std::string AffineTransformView::INITIAL_TRANSFORM_KEY     = "niftk.initaltransform";
const std::string AffineTransformView::INCREMENTAL_TRANSFORM_KEY = "niftk.incrementaltransform";
const std::string AffineTransformView::PRELOADED_TRANSFORM_KEY   = "niftk.preloadedtransform";
const std::string AffineTransformView::DISPLAYED_TRANSFORM_KEY   = "niftk.displayedtransform";
const std::string AffineTransformView::DISPLAYED_PARAMETERS_KEY  = "niftk.displayedtransformparameters";

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
}

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

void AffineTransformView::SetFocus()
{
}

void AffineTransformView::_SetControlsEnabled(bool isEnabled)
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

void AffineTransformView::_InitialiseTransformProperty(std::string name, mitk::DataNode& node)
{
  mitk::AffineTransformDataNodeProperty::Pointer transform
    = dynamic_cast<mitk::AffineTransformDataNodeProperty*>(node.GetProperty(name.c_str()));

  if (transform.IsNull())
  {
    transform = mitk::AffineTransformDataNodeProperty::New();
    transform->Identity();
    node.SetProperty(name.c_str(), transform);
  }
}

void AffineTransformView::_InitialiseNodeProperties(mitk::DataNode& node)
{
  // Make sure the node has the specified properties listed below, and if not create defaults.
  _InitialiseTransformProperty(INCREMENTAL_TRANSFORM_KEY, node);
  _InitialiseTransformProperty(PRELOADED_TRANSFORM_KEY, node);
  _InitialiseTransformProperty(DISPLAYED_TRANSFORM_KEY, node);

  mitk::AffineTransformParametersDataNodeProperty::Pointer affineTransformParametersProperty
    = dynamic_cast<mitk::AffineTransformParametersDataNodeProperty*>(node.GetProperty(DISPLAYED_PARAMETERS_KEY.c_str()));
  if (affineTransformParametersProperty.IsNull())
  {
    affineTransformParametersProperty = mitk::AffineTransformParametersDataNodeProperty::New();
    affineTransformParametersProperty->Identity();
    node.SetProperty(DISPLAYED_PARAMETERS_KEY.c_str(), affineTransformParametersProperty);
  }

  // In addition, if we have not already done so, we take any existing geometry,
  // and store it back on the node as the "Initial" geometry.
  mitk::AffineTransformDataNodeProperty::Pointer transform
    = dynamic_cast<mitk::AffineTransformDataNodeProperty*>(node.GetProperty(INITIAL_TRANSFORM_KEY.c_str()));
  if (transform.IsNull())
  {
    transform = mitk::AffineTransformDataNodeProperty::New();
    transform->SetTransform(*(const_cast<const vtkMatrix4x4*>(node.GetData()->GetGeometry()->GetVtkTransform()->GetMatrix())));
    node.SetProperty(INITIAL_TRANSFORM_KEY.c_str(), transform);
  }

  mitk::BaseData *data = node.GetData();
  if ( data == NULL )
  {
    MITK_ERROR << "No data object present!";
  }
}

void AffineTransformView::OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes)
{
  if (nodes.size() != 1)
  {
    this->_SetControlsEnabled(false);
    return;
  }

  if (nodes[0].IsNull())
  {
    this->_SetControlsEnabled(false);
    return;
  }

  // Store the current node as a member variable.
  msp_DataOwnerNode = nodes[0];

  // Initialise the selected node.
  this->_InitialiseNodeProperties(*(msp_DataOwnerNode.GetPointer()));
  
  // Initialise all of the selected nodes children.
  mitk::DataStorage::SetOfObjects::ConstPointer children = this->GetDataStorage()->GetDerivations(msp_DataOwnerNode.GetPointer());
  for (unsigned int i = 0; i < children->Size(); i++)
  {
    this->_InitialiseNodeProperties(*(children->GetElement(i)));
  }

  // Initialise the centre of rotation member variable.
  typedef itk::Point<mitk::ScalarType, 3> PointType;
  PointType centrePoint = msp_DataOwnerNode->GetData()->GetGeometry()->GetCenter();
  m_CentreOfRotation[0] = centrePoint[0];
  m_CentreOfRotation[1] = centrePoint[1];
  m_CentreOfRotation[2] = centrePoint[2];

  MITK_DEBUG << "OnSelectionChanged, set centre to " << m_CentreOfRotation[0] << ", " << m_CentreOfRotation[1] << ", " << m_CentreOfRotation[2] << std::endl;

  // Sets the GUI to the current transform parameters
  mitk::AffineTransformParametersDataNodeProperty::Pointer affineTransformParametersProperty
    = dynamic_cast<mitk::AffineTransformParametersDataNodeProperty*>(nodes[0]->GetProperty(DISPLAYED_PARAMETERS_KEY.c_str()));
  _SetControls(*(affineTransformParametersProperty.GetPointer()));

  _UpdateTransformDisplay();
  this->_SetControlsEnabled(true);

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

void AffineTransformView::_SetControls(mitk::AffineTransformParametersDataNodeProperty &parametersProperty)
{
  mitk::AffineTransformParametersDataNodeProperty::ParametersType params = parametersProperty.GetAffineTransformParameters();

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

void AffineTransformView::_GetControls(mitk::AffineTransformParametersDataNodeProperty &parametersProperty)
{
  mitk::AffineTransformParametersDataNodeProperty::ParametersType params = parametersProperty.GetAffineTransformParameters();

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

  parametersProperty.SetAffineTransformParameters(params);
}

void AffineTransformView::_ResetControls()
{
  mitk::AffineTransformParametersDataNodeProperty::Pointer affineTransformParametersProperty = mitk::AffineTransformParametersDataNodeProperty::New();
  affineTransformParametersProperty->Identity();
  _SetControls(*(affineTransformParametersProperty.GetPointer()));
}

vtkSmartPointer<vtkMatrix4x4> AffineTransformView::ComputeTransformFromParameters() const {

	vtkSmartPointer<vtkMatrix4x4> sp_inc, sp_tmp, sp_swap;
	double incVals[4][4], partInc[4][4], result[4][4];
	int cInd;

	vtkMatrix4x4::Identity(&incVals[0][0]);

	if (m_Controls->centreRotationRadioButton->isChecked()) {
		MITK_DEBUG << "Transform applied wrt. ("
				<< m_CentreOfRotation[0] << ", "
				<< m_CentreOfRotation[1] << ", "
				<< m_CentreOfRotation[2] << ")\n";

		for (cInd = 0; cInd < 3; cInd++)
			incVals[cInd][3] = -m_CentreOfRotation[cInd];
	}

	vtkMatrix4x4::Identity(&partInc[0][0]);
	partInc[0][0] = m_Controls->scalingSpinBoxX->value()/100.0;
	partInc[1][1] = m_Controls->scalingSpinBoxY->value()/100.0;
	partInc[2][2] = m_Controls->scalingSpinBoxZ->value()/100.0;
	vtkMatrix4x4::Multiply4x4(&partInc[0][0], &incVals[0][0], &result[0][0]);
	std::copy(&result[0][0], &result[0][0] + 16, &incVals[0][0]);

	vtkMatrix4x4::Identity(&partInc[0][0]);
	partInc[0][1] = m_Controls->shearSpinBoxXY->value();
	partInc[0][2] = m_Controls->shearSpinBoxXZ->value();
	partInc[1][2] = m_Controls->shearSpinBoxYZ->value();
	vtkMatrix4x4::Multiply4x4(&partInc[0][0], &incVals[0][0], &result[0][0]);
	std::copy(&result[0][0], &result[0][0] + 16, &incVals[0][0]);

	{
		double calpha, salpha, alpha;

		alpha = NIFTK_PI*m_Controls->rotationSpinBoxX->value()/180;
		calpha = cos(alpha);
		salpha = sin(alpha);

		vtkMatrix4x4::Identity(&partInc[0][0]);
		partInc[1][1] = calpha;
		partInc[1][2] = salpha;
		partInc[2][1] = -salpha;
		partInc[2][2] = calpha;
		vtkMatrix4x4::Multiply4x4(&partInc[0][0], &incVals[0][0], &result[0][0]);

		alpha = NIFTK_PI*m_Controls->rotationSpinBoxY->value()/180.0;
		calpha = cos(alpha);
		salpha = sin(alpha);

		vtkMatrix4x4::Identity(&partInc[0][0]);
		partInc[0][0] = calpha;
		partInc[0][2] = salpha;
		partInc[2][0] = -salpha;
		partInc[2][2] = calpha;
		vtkMatrix4x4::Multiply4x4(&partInc[0][0], &result[0][0], &incVals[0][0]);

		alpha = NIFTK_PI*m_Controls->rotationSpinBoxZ->value()/180.0;
		calpha = cos(alpha);
		salpha = sin(alpha);

		vtkMatrix4x4::Identity(&partInc[0][0]);
		partInc[0][0] = calpha;
		partInc[0][1] = salpha;
		partInc[1][0] = -salpha;
		partInc[1][1] = calpha;
		vtkMatrix4x4::Multiply4x4(&partInc[0][0], &incVals[0][0], &result[0][0]);

		std::copy(&result[0][0], &result[0][0] + 16, &incVals[0][0]);
	}

	incVals[0][3] += m_Controls->translationSpinBoxX->value();
	incVals[1][3] += m_Controls->translationSpinBoxY->value();
	incVals[2][3] += m_Controls->translationSpinBoxZ->value();

	if (m_Controls->centreRotationRadioButton->isChecked()) {
		for (cInd = 0; cInd < 3; cInd++) incVals[cInd][3] += m_CentreOfRotation[cInd];
	}

	sp_inc = vtkSmartPointer<vtkMatrix4x4>::New();
	std::copy(&incVals[0][0], &incVals[0][0] + 4*4, &sp_inc->Element[0][0]);

	return sp_inc;
}

void AffineTransformView::_UpdateTransformDisplay() {

  // This method gets a 4x4 matrix corresponding to the current transform, given by the current values
  // in all the rotation, translation, scaling and shearing widgets, and outputs the matrix in the GUI.
  // It does not actually change, or recompute, or transform anything. So we are just saying
  // "update the displayed view of the transformation".
	vtkSmartPointer<vtkMatrix4x4> sp_Transform = this->ComputeTransformFromParameters();
	for (int rInd = 0; rInd < 4; rInd++) for (int cInd = 0; cInd < 4; cInd++)
		m_Controls->affineTransformDisplay->setItem(rInd, cInd, new QTableWidgetItem(QString::number(sp_Transform->Element[rInd][cInd])));

}

void AffineTransformView::OnParameterChanged(const double) {
  _UpdateTransformDisplay();
  _UpdateTransformationGeometry();
}

void AffineTransformView::OnParameterChanged(const bool) {
  _UpdateTransformDisplay();
  _UpdateTransformationGeometry();
}

void AffineTransformView::_UpdateNodeProperties(
    const vtkSmartPointer<vtkMatrix4x4> displayedTransformFromParameters,
    const vtkSmartPointer<vtkMatrix4x4> incrementalTransformToBeComposed,
    mitk::DataNode& node)
{
  _UpdateTransformProperty(DISPLAYED_TRANSFORM_KEY, displayedTransformFromParameters, node);
  _UpdateTransformProperty(INCREMENTAL_TRANSFORM_KEY, incrementalTransformToBeComposed, node);

  // Get the parameters from the controls, and store on node.
  mitk::AffineTransformParametersDataNodeProperty::Pointer affineTransformParametersProperty = mitk::AffineTransformParametersDataNodeProperty::New();
  this->_GetControls(*affineTransformParametersProperty);
  node.ReplaceProperty(DISPLAYED_PARAMETERS_KEY.c_str(), affineTransformParametersProperty);

  // Compose the transform with the current geometry, and force modified flags to make sure we get a re-rendering.
  node.GetData()->GetGeometry()->Compose( incrementalTransformToBeComposed );
  node.GetData()->Modified();
  node.Modified();
}

void AffineTransformView::_UpdateTransformProperty(std::string name, vtkSmartPointer<vtkMatrix4x4> transform, mitk::DataNode& node)
{
  mitk::AffineTransformDataNodeProperty::Pointer property = mitk::AffineTransformDataNodeProperty::New();
  property->SetTransform((*(transform.GetPointer())));
  node.ReplaceProperty(name.c_str(), property);
}

void AffineTransformView::OnResetTransformPushed() {
  _ResetControls();
  _UpdateTransformDisplay();
  _UpdateTransformationGeometry();
}

void AffineTransformView::_UpdateTransformationGeometry()
{
  /**************************************************************
   * This is the main method composing and calculating matrices.
   **************************************************************/
  if (msp_DataOwnerNode.IsNotNull())
  {
    vtkSmartPointer<vtkMatrix4x4> sp_TransformDisplayed = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(DISPLAYED_TRANSFORM_KEY.c_str(), *(msp_DataOwnerNode.GetPointer()));
    vtkSmartPointer<vtkMatrix4x4> sp_TransformPreLoaded = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(PRELOADED_TRANSFORM_KEY.c_str(), *(msp_DataOwnerNode.GetPointer()));

    vtkSmartPointer<vtkMatrix4x4> sp_InvertedDisplayedTransform = vtkMatrix4x4::New();
    vtkMatrix4x4::Invert(sp_TransformDisplayed, sp_InvertedDisplayedTransform);

    vtkSmartPointer<vtkMatrix4x4> sp_InvertedTransformPreLoaded = vtkMatrix4x4::New();
    vtkMatrix4x4::Invert(sp_TransformPreLoaded, sp_InvertedTransformPreLoaded);

    vtkSmartPointer<vtkMatrix4x4> sp_NewTransformAccordingToParameters = this->ComputeTransformFromParameters();

    vtkSmartPointer<vtkMatrix4x4> sp_InvertedTransforms = vtkMatrix4x4::New();
    vtkSmartPointer<vtkMatrix4x4> sp_TransformsBeforeAffine = vtkMatrix4x4::New();
    vtkSmartPointer<vtkMatrix4x4> sp_FinalAffineTransform = vtkMatrix4x4::New();

    vtkMatrix4x4::Multiply4x4(sp_InvertedTransformPreLoaded, sp_InvertedDisplayedTransform, sp_InvertedTransforms);
    vtkMatrix4x4::Multiply4x4(sp_TransformPreLoaded, sp_InvertedTransforms, sp_TransformsBeforeAffine);
    vtkMatrix4x4::Multiply4x4(sp_NewTransformAccordingToParameters, sp_TransformsBeforeAffine, sp_FinalAffineTransform);

    this->_UpdateNodeProperties(
        sp_NewTransformAccordingToParameters,
        sp_FinalAffineTransform,
        *(msp_DataOwnerNode.GetPointer())
        );

    mitk::DataStorage::SetOfObjects::ConstPointer children = this->GetDataStorage()->GetDerivations(msp_DataOwnerNode.GetPointer());
    for (unsigned int i = 0; i < children->Size(); i++)
    {
      this->_UpdateNodeProperties(
          sp_NewTransformAccordingToParameters,
          sp_FinalAffineTransform,
          *(children->GetElement(i))
          );
    }

    QmitkAbstractView::RequestRenderWindowUpdate();
  }
}

void AffineTransformView::_ApplyLoadedTransformToNode(
    const vtkSmartPointer<vtkMatrix4x4> transformFromFile,
    mitk::DataNode& node)
{
  /**************************************************************
   * This is the main method to apply a transformation from file.
   **************************************************************/

  vtkSmartPointer<vtkMatrix4x4> incrementalTransformation = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(INCREMENTAL_TRANSFORM_KEY.c_str(), node);
  vtkSmartPointer<vtkMatrix4x4> invertedIncrementalTransformation = vtkMatrix4x4::New();
  vtkMatrix4x4::Invert(incrementalTransformation, invertedIncrementalTransformation);
  node.GetData()->GetGeometry()->Compose( invertedIncrementalTransformation );

  mitk::AffineTransformDataNodeProperty::Pointer transformFromFileProperty = mitk::AffineTransformDataNodeProperty::New();
  transformFromFileProperty->SetTransform(*(transformFromFile.GetPointer()));
  node.ReplaceProperty(PRELOADED_TRANSFORM_KEY.c_str(), transformFromFileProperty);
  node.GetData()->GetGeometry()->Compose( transformFromFile );

  mitk::AffineTransformDataNodeProperty::Pointer affineTransformIdentity = mitk::AffineTransformDataNodeProperty::New();
  affineTransformIdentity->Identity();
  node.ReplaceProperty(DISPLAYED_TRANSFORM_KEY.c_str(), affineTransformIdentity);

  mitk::AffineTransformParametersDataNodeProperty::Pointer affineTransformParametersIdentity = mitk::AffineTransformParametersDataNodeProperty::New();
  affineTransformParametersIdentity->Identity();
  node.ReplaceProperty(DISPLAYED_PARAMETERS_KEY.c_str(), affineTransformParametersIdentity);
}

template <const unsigned int t_Dim>
static vtkSmartPointer<vtkMatrix4x4> _ConvertFromITKTransform(const itk::TransformBase &itkTransform) throw (itk::ExceptionObject){
	typedef itk::AffineTransform<double, t_Dim> __ITKTransformType;

	vtkSmartPointer<vtkMatrix4x4> sp_mat;

	sp_mat = vtkSmartPointer<vtkMatrix4x4>::New();
	sp_mat->Identity();

	if (std::string(itkTransform.GetNameOfClass()) == "AffineTransform" && itkTransform.GetOutputSpaceDimension() == t_Dim) {
		const __ITKTransformType &itkAffineTransform = *static_cast<const __ITKTransformType*>(&itkTransform);
		const typename __ITKTransformType::MatrixType matrix = itkAffineTransform.GetMatrix();
		const typename __ITKTransformType::OutputVectorType trans = itkAffineTransform.GetOffset();

		unsigned int rInd, cInd;

		MITK_DEBUG << "Reading transform:\n";
		for (rInd = 0; rInd < t_Dim; rInd++) {
			for (cInd = 0; cInd < t_Dim; cInd++) {
				MITK_DEBUG << matrix(rInd,cInd) << " ";
				sp_mat->Element[rInd][cInd] = matrix(rInd,cInd);
			}

			sp_mat->Element[rInd][3] = trans[rInd];
			MITK_DEBUG << trans[rInd] << std::endl;
		}
	} else {
		itkGenericExceptionMacro(<< "Failed to cast input transform to ITK affine transform.\nInput transform has type " << itkTransform.GetNameOfClass() << " (" << itkTransform.GetOutputSpaceDimension() << "D)\n");
	}

	return sp_mat;
}

void AffineTransformView::OnLoadTransformPushed() {

  assert(msp_DataOwnerNode);

  QString fileName;
	fileName = QFileDialog::getOpenFileName(NULL, tr("Select transform file"), QString(), tr("ITK affine transform file (*.txt *.tfm);;Any file (*)"));
	if (fileName.length() > 0) {

		itk::TransformFileReader::Pointer sp_transformIO;

		try {
			sp_transformIO = itk::TransformFileReader::New();
			sp_transformIO->SetFileName(fileName.toStdString().c_str());
			sp_transformIO->Update();

			if (sp_transformIO->GetTransformList()->size() == 0) {
				MITK_ERROR << "ITK didn't find any transforms in " << fileName.toStdString() << endl;
				return;
			}
			vtkSmartPointer<vtkMatrix4x4> transformFromFile = _ConvertFromITKTransform<3> (*sp_transformIO->GetTransformList()->front());
			MITK_DEBUG << "Reading of transform from file: success";

			this->_ApplyLoadedTransformToNode(transformFromFile, *(msp_DataOwnerNode.GetPointer()));

			mitk::DataStorage::SetOfObjects::ConstPointer children = this->GetDataStorage()->GetDerivations(msp_DataOwnerNode.GetPointer());
			for (unsigned int i = 0; i < children->Size(); i++) {
				this->_ApplyLoadedTransformToNode(transformFromFile, *(children->GetElement(i)));
			}

			OnResetTransformPushed();

			MITK_DEBUG << "Applied transform from file: success";

		} catch (itk::ExceptionObject &r_itkEx) {
			MITK_ERROR << "Transform " << fileName.toStdString()
					<< " is incompatible with image.\n"
					<< "Caught ITK exception:\n" << r_itkEx.what() << std::endl;
		}
	}
}

template <const unsigned int t_Dim, const bool t_DoInvert>
static typename itk::AffineTransform<double, t_Dim>::Pointer _ConvertToITKTransform(const vtkMatrix4x4 &transform) {
  typedef itk::AffineTransform<double, t_Dim> __ITKTransform;

  typename __ITKTransform::Pointer sp_itkTransform;
  typename __ITKTransform::MatrixType itkMatrix;
  typename  __ITKTransform::OutputVectorType itkVec;
  unsigned int rInd, cInd;
  vtkSmartPointer<vtkMatrix4x4> sp_vtkInptTransform;

  /*
  * -Invert input transform since itk resampler expects an inverse transform that can be applied to the coordinates.
  */
  sp_itkTransform = __ITKTransform::New();
  sp_vtkInptTransform =vtkSmartPointer<vtkMatrix4x4>::New();

  if (t_DoInvert) {
	  vtkMatrix4x4::Invert(const_cast<vtkMatrix4x4*>(&transform), sp_vtkInptTransform);
  } else {
	  sp_vtkInptTransform->DeepCopy(const_cast<vtkMatrix4x4*>(&transform));
  }

  MITK_DEBUG << "Converting transform " << std::endl;
  //sp_vtkInptTransform->PrintSelf(MITK_DEBUG, *vtkIndent::New());
  MITK_DEBUG << "to ITK\n";

  for (rInd = 0; rInd < t_Dim; rInd++) {
    for (cInd = 0; cInd < t_Dim; cInd++) itkMatrix(rInd,cInd) = sp_vtkInptTransform->Element[rInd][cInd];
    itkVec[rInd] = sp_vtkInptTransform->Element[rInd][3];
  }

  sp_itkTransform->SetMatrix(itkMatrix);
  sp_itkTransform->SetOffset(itkVec);

  return sp_itkTransform;
}

void AffineTransformView::OnSaveTransformPushed() {

  assert(msp_DataOwnerNode);

  QString fileName;
  fileName = QFileDialog::getSaveFileName(NULL, tr("Destination for transform"), QString(), tr("ITK affine transform file (*.tfm *.txt);;Any file (*)"));
  if (fileName.length() > 0) {

    itk::TransformFileWriter::Pointer sp_writer;
    sp_writer = itk::TransformFileWriter::New();
    sp_writer->SetFileName(fileName.toStdString().c_str());

    try {

      vtkSmartPointer<vtkMatrix4x4> transform = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(DISPLAYED_TRANSFORM_KEY.c_str(), *(msp_DataOwnerNode.GetPointer()));
      sp_writer->SetInput(_ConvertToITKTransform<3, false>(*transform));
      sp_writer->Update();

      MITK_DEBUG << "Writing of current transform to file: success";
    } catch (itk::ExceptionObject &r_ex) {
      MITK_ERROR << "Caught ITK exception:\n"
          << r_ex.what() << endl;
    }
  }
}

template <typename TPixelType, unsigned int t_Dim>
void _ApplyTransform(itk::Image<TPixelType, t_Dim> *p_itkImg, const vtkMatrix4x4 &transform) {
	typedef itk::Image<TPixelType, t_Dim> __ITKImageType;

	mitk::Image::Pointer sp_transImg;
	typename itk::ResampleImageFilter<__ITKImageType, __ITKImageType>::Pointer sp_resampler;
	typename itk::AffineTransform<double, t_Dim>::Pointer sp_itkTransform;

	sp_resampler = itk::ResampleImageFilter<__ITKImageType, __ITKImageType>::New();
	sp_itkTransform = _ConvertToITKTransform<t_Dim, true>(transform);
	sp_resampler->SetTransform(sp_itkTransform);

	sp_resampler->SetInput(p_itkImg);
	sp_resampler->SetInterpolator(itk::LinearInterpolateImageFunction<__ITKImageType>::New());

	sp_resampler->SetUseReferenceImage(true);
	sp_resampler->SetReferenceImage(p_itkImg);

	try {
		sp_resampler->UpdateLargestPossibleRegion();

    typename itk::ImageRegionConstIterator<__ITKImageType> resampledIterator(sp_resampler->GetOutput(), sp_resampler->GetOutput()->GetLargestPossibleRegion());
    typename itk::ImageRegionIterator<__ITKImageType> inputIterator(p_itkImg, sp_resampler->GetOutput()->GetLargestPossibleRegion());
    for (resampledIterator.GoToBegin(), inputIterator.GoToBegin();
        !resampledIterator.IsAtEnd();
        ++resampledIterator, ++inputIterator)
    {
      inputIterator.Set(resampledIterator.Get());
    }
	} catch (itk::ExceptionObject &r_itkEx) {
		MITK_ERROR << r_itkEx.what() << std::endl;
		return;
	}

	MITK_DEBUG << "Processing: success\n";
}

template <typename TMultiChannelPixelType, unsigned int t_Dim>
void _ApplyTransformMultiChannel(itk::Image<TMultiChannelPixelType, t_Dim> *p_itkImg, const vtkMatrix4x4 &transform) {
	typedef itk::Image<TMultiChannelPixelType, t_Dim> __ITKImageType;

	mitk::Image::Pointer sp_transImg;
	typename itk::VectorResampleImageFilter<__ITKImageType, __ITKImageType>::Pointer sp_resampler;
	typename itk::AffineTransform<double, t_Dim>::Pointer sp_itkTransform;

	sp_resampler = itk::VectorResampleImageFilter<__ITKImageType, __ITKImageType>::New();
	sp_itkTransform = _ConvertToITKTransform<t_Dim, true>(transform);
	sp_resampler->SetTransform(sp_itkTransform);

	sp_resampler->SetInput(p_itkImg);
	sp_resampler->SetInterpolator(itk::VectorLinearInterpolateImageFunction<__ITKImageType>::New());

	sp_resampler->SetSize(p_itkImg->GetLargestPossibleRegion().GetSize());
	sp_resampler->SetOutputSpacing(p_itkImg->GetSpacing());
	sp_resampler->SetOutputDirection(p_itkImg->GetDirection());
	sp_resampler->SetOutputOrigin(p_itkImg->GetOrigin());

	try {
		sp_resampler->UpdateLargestPossibleRegion();

		typename itk::ImageRegionConstIterator<__ITKImageType> resampledIterator(sp_resampler->GetOutput(), sp_resampler->GetOutput()->GetLargestPossibleRegion());
		typename itk::ImageRegionIterator<__ITKImageType> inputIterator(p_itkImg, sp_resampler->GetOutput()->GetLargestPossibleRegion());
		for (resampledIterator.GoToBegin(), inputIterator.GoToBegin();
		    !resampledIterator.IsAtEnd();
		    ++resampledIterator, ++inputIterator)
		{
		  inputIterator.Set(resampledIterator.Get());
		}

	} catch (itk::ExceptionObject &r_itkEx) {
		MITK_ERROR << r_itkEx.what() << std::endl;
		return;
	}

	MITK_DEBUG << "Processing: success\n";
}

void AffineTransformView::_ApplyResampleToCurrentNode() {

  assert(msp_DataOwnerNode.IsNotNull());
 
  mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(msp_DataOwnerNode->GetData());
  assert(image);

  vtkSmartPointer<vtkMatrix4x4> sp_transformFromParams = this->ComputeTransformFromParameters();
  vtkSmartPointer<vtkMatrix4x4> sp_transformPreLoaded = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(PRELOADED_TRANSFORM_KEY.c_str(), *(msp_DataOwnerNode.GetPointer()));
  vtkSmartPointer<vtkMatrix4x4> sp_combinedTransform = vtkMatrix4x4::New();
  vtkMatrix4x4::Multiply4x4(sp_transformFromParams, sp_transformPreLoaded, sp_combinedTransform);
  vtkSmartPointer<vtkMatrix4x4> sp_incTransform = vtkMatrix4x4::New();
  vtkMatrix4x4::Invert(sp_combinedTransform, sp_incTransform);

  try {
#define APPLY_MULTICHANNEL(TMultiChannelType) \
      AccessFixedPixelTypeByItk_n(image, _ApplyTransformMultiChannel, ( TMultiChannelType ), (*sp_combinedTransform))

    if (image->GetPixelType().GetNumberOfComponents() == 3) {
#define APPLY_MULTICHANNEL_RGB(TBaseType) APPLY_MULTICHANNEL(itk::RGBPixel< TBaseType >)

      if (image->GetPixelType().GetTypeId() == typeid(signed char) && image->GetPixelType().GetBitsPerComponent() == 8) {
        MITK_DEBUG << "Assuming RGB (signed char)\n"
            << "ITK typeID: " << typeid(itk::RGBPixel<signed char>).name() << std::endl;

        APPLY_MULTICHANNEL_RGB(signed char);
      } else if (image->GetPixelType().GetTypeId() == typeid(unsigned char) && image->GetPixelType().GetBitsPerComponent() == 8) {
        MITK_DEBUG << "Assuming RGB (unsigned char)\n"
            << "ITK typeID: " << typeid(itk::RGBPixel<unsigned char>).name() << std::endl;

        APPLY_MULTICHANNEL_RGB(unsigned char);
      } else if (image->GetPixelType().GetTypeId() == typeid(signed short) && image->GetPixelType().GetBitsPerComponent() == 16) {
        APPLY_MULTICHANNEL_RGB(signed short);
      } else if (image->GetPixelType().GetTypeId() == typeid(unsigned short) && image->GetPixelType().GetBitsPerComponent() == 16) {
        APPLY_MULTICHANNEL_RGB(unsigned short);
      } else if (image->GetPixelType().GetTypeId() == typeid(float) && image->GetPixelType().GetBitsPerComponent() == 32) {
        MITK_DEBUG << "Assuming RGB (float)\n"
            << "ITK typeID: " << typeid(itk::RGBPixel<float>).name() << std::endl;

        APPLY_MULTICHANNEL_RGB(float);
      } else if (image->GetPixelType().GetTypeId()== typeid(double) && image->GetPixelType().GetBitsPerComponent() == 64) {
        APPLY_MULTICHANNEL_RGB(double);
      } else {
        MITK_ERROR << "pixel type " << image->GetPixelType().GetItkTypeAsString() << " is not supported.\n";
      }
#undef APPLY_MULTICHANNEL_RGB
    } else if (image->GetPixelType().GetNumberOfComponents() == 4) {
#define APPLY_MULTICHANNEL_RGBA(TBaseType) APPLY_MULTICHANNEL(itk::RGBAPixel< TBaseType >)

      if (image->GetPixelType().GetTypeId() == typeid(unsigned char) && image->GetPixelType().GetBitsPerComponent() == 8) {
        MITK_DEBUG << "Assuming RGB (unsigned char)\n"
            << "ITK typeID: " << typeid(itk::RGBAPixel<unsigned char>).name() << std::endl;

        APPLY_MULTICHANNEL_RGBA(signed char);
      } else if (image->GetPixelType().GetTypeId() == typeid(signed char) && image->GetPixelType().GetBitsPerComponent() == 8) {
        MITK_DEBUG << "Assuming RGB (signed char)\n"
            << "ITK typeID: " << typeid(itk::RGBAPixel<signed char>).name() << std::endl;

        APPLY_MULTICHANNEL_RGBA(unsigned char);
      } else if (image->GetPixelType().GetTypeId() == typeid(signed short) && image->GetPixelType().GetBitsPerComponent() == 16) {
        APPLY_MULTICHANNEL_RGBA(signed short);
      } else if (image->GetPixelType().GetTypeId() == typeid(unsigned short) && image->GetPixelType().GetBitsPerComponent() == 16) {
        APPLY_MULTICHANNEL_RGBA(unsigned short);
      } else if (image->GetPixelType().GetTypeId() == typeid(float) && image->GetPixelType().GetBitsPerComponent() == 32) {
        MITK_DEBUG << "Assuming RGB (float)\n"
            << "ITK typeID: " << typeid(itk::RGBAPixel<float>).name() << std::endl;

        APPLY_MULTICHANNEL_RGBA(float);
      } else if (image->GetPixelType().GetTypeId()== typeid(double) && image->GetPixelType().GetBitsPerComponent() == 64) {
        APPLY_MULTICHANNEL_RGBA(double);
      } else
        MITK_ERROR << "pixel type " << image->GetPixelType().GetItkTypeAsString() << " is not supported.\n";

#undef APPLY_MULTICHANNEL_RGBA
    } else {
      AccessByItk_n(image, _ApplyTransform, (*sp_combinedTransform));
    }

#undef APPLY_MULTICHANNEL
  } catch (mitk::AccessByItkException &r_ex) {
    MITK_ERROR << "MITK Exception:\n" << r_ex.what() << endl;
  }

  image->Modified();
  msp_DataOwnerNode->Modified();
}

void AffineTransformView::OnResampleTransformPushed() {

  assert(msp_DataOwnerNode.IsNotNull());

  mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(msp_DataOwnerNode->GetData());
  assert(image);

  if (image.IsNotNull())
  {
    std::stringstream message;
    std::string name;
    message << "Performing image processing for image ";
    if (msp_DataOwnerNode->GetName(name)) {
      // a property called "name" was found for this DataNode
      message << "'" << name << "'";
    }
    message << ".";
    MITK_DEBUG << message.str();

    // Reset the geometry, in a similar fashion to when we load a new transformation.
    vtkSmartPointer<vtkMatrix4x4> total = msp_DataOwnerNode->GetData()->GetGeometry()->GetVtkTransform()->GetMatrix();
    vtkSmartPointer<vtkMatrix4x4> totalInverted = vtkMatrix4x4::New();
    vtkMatrix4x4::Invert(total, totalInverted);
    msp_DataOwnerNode->GetData()->GetGeometry()->Compose( totalInverted );

    vtkSmartPointer<vtkMatrix4x4> initial = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(INITIAL_TRANSFORM_KEY.c_str(), *(msp_DataOwnerNode.GetPointer()));
    msp_DataOwnerNode->GetData()->GetGeometry()->Compose( initial );

    // Do the resampling, according to current GUI parameters, which represent the "current" transformation.
    _ApplyResampleToCurrentNode();

    vtkSmartPointer<vtkMatrix4x4> identity = vtkMatrix4x4::New();
    identity->Identity();
    mitk::AffineTransformDataNodeProperty::StoreTransformInNode(INCREMENTAL_TRANSFORM_KEY, *(identity.GetPointer()), *(msp_DataOwnerNode.GetPointer()));
    mitk::AffineTransformDataNodeProperty::StoreTransformInNode(PRELOADED_TRANSFORM_KEY, *(identity.GetPointer()), *(msp_DataOwnerNode.GetPointer()));
    mitk::AffineTransformDataNodeProperty::StoreTransformInNode(DISPLAYED_TRANSFORM_KEY, *(identity.GetPointer()), *(msp_DataOwnerNode.GetPointer()));

    // Then reset the parameters.
    _ResetControls();
    _UpdateTransformDisplay();
  }
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
