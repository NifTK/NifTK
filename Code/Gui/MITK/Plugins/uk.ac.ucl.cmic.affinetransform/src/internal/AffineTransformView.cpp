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

// VTK
#include "vtkLinearTransform.h"
#include "vtkMatrix4x4.h"

// MITK Misc
#include <mitkImage.h>
#include <mitkImageAccessByItk.h>
#include <mitkITKImageImport.h>
#include <mitkInstantiateAccessFunctions.h>
#include <mitkVector.h> // for PointType;
#include <mitkIDataStorageService.h>
#include <mitkRenderingManager.h>

// NIFTK
#include "AffineTransformView.h"
#include "mitkAffineTransformDataNodeProperty.h"
#include "mitkAffineTransformParametersDataNodeProperty.h"
#include "ConversionUtils.h"

#include <itkImageFileWriter.h>

const std::string AffineTransformView::VIEW_ID = "uk.ac.ucl.cmic.affinetransformview";
const std::string AffineTransformView::INITIAL_TRANSFORM_KEY = "niftk.initaltransform";
const std::string AffineTransformView::PRELOADED_TRANSFORM_KEY = "niftk.preloadedtransform";
const std::string AffineTransformView::COMBINED_TRANSFORM_KEY = "niftk.combinedtransform";

AffineTransformView::AffineTransformView()
:
  m_Controls(NULL)
, msp_DataOwnerNode(NULL)
{
}

AffineTransformView::~AffineTransformView()
{
  if (m_Controls != NULL)
  {
    delete m_Controls;
  }
}

void AffineTransformView::CreateQtPartControl( QWidget *parent )
{
  if (!m_Controls)
  {
    // create GUI widgets from the Qt Designer's .ui file
    m_Controls = new Ui::AffineTransformWidget();
    m_Controls->setupUi(parent);

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
  }
}

void AffineTransformView::SetFocus()
{
  m_Controls->resetButton->setFocus();
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
  _InitialiseTransformProperty(PRELOADED_TRANSFORM_KEY, node);
  _InitialiseTransformProperty(COMBINED_TRANSFORM_KEY, node);
  _InitialiseTransformProperty(mitk::AffineTransformDataNodeProperty::PropertyKey, node);

  mitk::AffineTransformParametersDataNodeProperty::Pointer affineTransformParametersProperty
    = dynamic_cast<mitk::AffineTransformParametersDataNodeProperty*>(node.GetProperty(mitk::AffineTransformParametersDataNodeProperty::PropertyKey.c_str()));
  if (affineTransformParametersProperty.IsNull())
  {
    affineTransformParametersProperty = mitk::AffineTransformParametersDataNodeProperty::New();
    affineTransformParametersProperty->Identity();
    node.SetProperty(mitk::AffineTransformParametersDataNodeProperty::PropertyKey.c_str(), affineTransformParametersProperty);
  }

  mitk::AffineTransformDataNodeProperty::Pointer initialTransformProperty
    = dynamic_cast<mitk::AffineTransformDataNodeProperty*>(node.GetProperty(INITIAL_TRANSFORM_KEY.c_str()));
  if (initialTransformProperty.IsNull())
  {
    initialTransformProperty = mitk::AffineTransformDataNodeProperty::New();
    initialTransformProperty->SetTransform(*(node.GetData()->GetGeometry()->GetVtkTransform()->GetMatrix()));
    node.SetProperty(INITIAL_TRANSFORM_KEY.c_str(), initialTransformProperty);
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
    = dynamic_cast<mitk::AffineTransformParametersDataNodeProperty*>(nodes[0]->GetProperty(mitk::AffineTransformParametersDataNodeProperty::PropertyKey.c_str()));
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

void AffineTransformView::_UpdateTransformProperty(std::string name, vtkSmartPointer<vtkMatrix4x4> transform, mitk::DataNode& node)
{
  mitk::AffineTransformDataNodeProperty::Pointer property = mitk::AffineTransformDataNodeProperty::New();
  property->SetTransform((*(transform.GetPointer())));
  node.ReplaceProperty(name.c_str(), property);
}

void AffineTransformView::_UpdateNodeProperties(
    const vtkSmartPointer<vtkMatrix4x4> transformFromParameters,
    const vtkSmartPointer<vtkMatrix4x4> combinedTransform,
    mitk::DataNode& node)
{
  _UpdateTransformProperty(mitk::AffineTransformDataNodeProperty::PropertyKey, transformFromParameters, node);
  _UpdateTransformProperty(COMBINED_TRANSFORM_KEY, combinedTransform, node);

  // Get the parameters from the controls, and store on node.
  mitk::AffineTransformParametersDataNodeProperty::Pointer affineTransformParametersProperty = mitk::AffineTransformParametersDataNodeProperty::New();
  this->_GetControls(*affineTransformParametersProperty);
  node.ReplaceProperty(mitk::AffineTransformParametersDataNodeProperty::PropertyKey.c_str(), affineTransformParametersProperty);

  // Compose the transform with the current geometry, and force modified flags to make sure we get a re-rendering.
  node.GetData()->GetGeometry()->Compose( combinedTransform );
  node.GetData()->Modified();
  node.Modified();
}

void AffineTransformView::_UpdateTransformationGeometry()
{
  if (msp_DataOwnerNode.IsNotNull())
  {
    vtkSmartPointer<vtkMatrix4x4> sp_TransformFromParameters = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(mitk::AffineTransformDataNodeProperty::PropertyKey.c_str(), *(msp_DataOwnerNode.GetPointer()));
    vtkSmartPointer<vtkMatrix4x4> sp_TransformPreLoaded = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(PRELOADED_TRANSFORM_KEY.c_str(), *(msp_DataOwnerNode.GetPointer()));

    vtkSmartPointer<vtkMatrix4x4> sp_InvertedTransformFromParameters = vtkMatrix4x4::New();
    vtkMatrix4x4::Invert(sp_TransformFromParameters, sp_InvertedTransformFromParameters);

    vtkSmartPointer<vtkMatrix4x4> sp_InvertedTransformPreLoaded = vtkMatrix4x4::New();
    vtkMatrix4x4::Invert(sp_TransformPreLoaded, sp_InvertedTransformPreLoaded);

    vtkSmartPointer<vtkMatrix4x4> sp_NewTransformFromParameters = this->ComputeTransformFromParameters();

    vtkSmartPointer<vtkMatrix4x4> sp_InvertedTransforms = vtkMatrix4x4::New();
    vtkSmartPointer<vtkMatrix4x4> sp_TransformsBeforeAffine = vtkMatrix4x4::New();

    vtkSmartPointer<vtkMatrix4x4> sp_FinalAffineTransform = vtkMatrix4x4::New();

    vtkMatrix4x4::Multiply4x4(sp_InvertedTransformPreLoaded, sp_InvertedTransformFromParameters, sp_InvertedTransforms);
    vtkMatrix4x4::Multiply4x4(sp_TransformPreLoaded, sp_InvertedTransforms, sp_TransformsBeforeAffine);
    vtkMatrix4x4::Multiply4x4(sp_NewTransformFromParameters, sp_TransformsBeforeAffine, sp_FinalAffineTransform);

    this->_UpdateNodeProperties(
        sp_NewTransformFromParameters,
        sp_FinalAffineTransform,
        *(msp_DataOwnerNode.GetPointer())
        );

    mitk::DataStorage::SetOfObjects::ConstPointer children = this->GetDataStorage()->GetDerivations(msp_DataOwnerNode.GetPointer());
    for (unsigned int i = 0; i < children->Size(); i++)
    {
      this->_UpdateNodeProperties(
          sp_NewTransformFromParameters,
          sp_FinalAffineTransform,
          *(children->GetElement(i))
          );
    }

    QmitkAbstractView::RequestRenderWindowUpdate();
  }
}

void AffineTransformView::OnResetTransformPushed() {
	_ResetControls();
	_UpdateTransformDisplay();
	_UpdateTransformationGeometry();
}

void AffineTransformView::_ApplyLoadedTransformToNode(
    const vtkSmartPointer<vtkMatrix4x4> transformFromFile,
    mitk::DataNode& node)
{
  vtkSmartPointer<vtkMatrix4x4> combinedTransformation = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(COMBINED_TRANSFORM_KEY.c_str(), node);
  vtkSmartPointer<vtkMatrix4x4> invertedCombinedTransformation = vtkMatrix4x4::New();
  vtkMatrix4x4::Invert(combinedTransformation, invertedCombinedTransformation);
  node.GetData()->GetGeometry()->Compose( invertedCombinedTransformation );

  mitk::AffineTransformDataNodeProperty::Pointer transformFromFileProperty = mitk::AffineTransformDataNodeProperty::New();
  transformFromFileProperty->SetTransform(*(transformFromFile.GetPointer()));
  node.ReplaceProperty(PRELOADED_TRANSFORM_KEY.c_str(), transformFromFileProperty);
  node.GetData()->GetGeometry()->Compose( transformFromFile );

  mitk::AffineTransformDataNodeProperty::Pointer affineTransformIdentity = mitk::AffineTransformDataNodeProperty::New();
  affineTransformIdentity->Identity();
  node.ReplaceProperty(mitk::AffineTransformDataNodeProperty::PropertyKey.c_str(), affineTransformIdentity);

  mitk::AffineTransformParametersDataNodeProperty::Pointer affineTransformParametersIdentity = mitk::AffineTransformParametersDataNodeProperty::New();
  affineTransformParametersIdentity->Identity();
  node.ReplaceProperty(mitk::AffineTransformParametersDataNodeProperty::PropertyKey.c_str(), affineTransformParametersIdentity);
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

      vtkSmartPointer<vtkMatrix4x4> transform = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(mitk::AffineTransformDataNodeProperty::PropertyKey.c_str(), *(msp_DataOwnerNode.GetPointer()));
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

/** TODO: Matt says: mitkIpPic has been deprecated, so this needs reworking
  try {
#define APPLY_MULTICHANNEL(TMultiChannelType) \
      AccessFixedPixelTypeByItk_n(image, _ApplyTransformMultiChannel, ( TMultiChannelType ), (*sp_incTransform))

    if (image->GetPixelType().GetNumberOfComponents() == 3) {
#define APPLY_MULTICHANNEL_RGB(TBaseType) APPLY_MULTICHANNEL(itk::RGBPixel< TBaseType >)

      if (image->GetPixelType().GetType() == mitkIpPicInt && image->GetPixelType().GetBitsPerComponent() == 8) {
        MITK_DEBUG << "Assuming RGB (unsigned char)\n"
            << "ITK typeID: " << typeid(itk::RGBPixel<unsigned char>).name() << std::endl;

        APPLY_MULTICHANNEL_RGB(signed char);
      } else if (image->GetPixelType().GetType() == mitkIpPicUInt && image->GetPixelType().GetBitsPerComponent() == 8) {
        MITK_DEBUG << "Assuming RGB (signed char)\n"
            << "ITK typeID: " << typeid(itk::RGBPixel<signed char>).name() << std::endl;

        APPLY_MULTICHANNEL_RGB(unsigned char);
      } else if (image->GetPixelType().GetType() == mitkIpPicInt && image->GetPixelType().GetBitsPerComponent() == 16) {
        APPLY_MULTICHANNEL_RGB(signed short);
      } else if (image->GetPixelType().GetType() == mitkIpPicUInt && image->GetPixelType().GetBitsPerComponent() == 16) {
        APPLY_MULTICHANNEL_RGB(unsigned short);
      } else if (image->GetPixelType().GetType() == mitkIpPicFloat && image->GetPixelType().GetBitsPerComponent() == 32) {
        MITK_DEBUG << "Assuming RGB (float)\n"
            << "ITK typeID: " << typeid(itk::RGBPixel<float>).name() << std::endl;

        APPLY_MULTICHANNEL_RGB(float);
      } else if (image->GetPixelType().GetType()== mitkIpPicFloat && image->GetPixelType().GetBitsPerComponent() == 64) {
        APPLY_MULTICHANNEL_RGB(double);
      } else
        MITK_ERROR << "pixel type " << image->GetPixelType().GetItkTypeAsString() << " is not supported.\n";

#undef APPLY_MULTICHANNEL_RGB
    } else if (image->GetPixelType().GetNumberOfComponents() == 4) {
#define APPLY_MULTICHANNEL_RGBA(TBaseType) APPLY_MULTICHANNEL(itk::RGBAPixel< TBaseType >)

      if (image->GetPixelType().GetType() == mitkIpPicInt && image->GetPixelType().GetBitsPerComponent() == 8) {
        MITK_DEBUG << "Assuming RGB (unsigned char)\n"
            << "ITK typeID: " << typeid(itk::RGBAPixel<unsigned char>).name() << std::endl;

        APPLY_MULTICHANNEL_RGBA(signed char);
      } else if (image->GetPixelType().GetType() == mitkIpPicUInt && image->GetPixelType().GetBitsPerComponent() == 8) {
        MITK_DEBUG << "Assuming RGB (signed char)\n"
            << "ITK typeID: " << typeid(itk::RGBAPixel<signed char>).name() << std::endl;

        APPLY_MULTICHANNEL_RGBA(unsigned char);
      } else if (image->GetPixelType().GetType() == mitkIpPicInt && image->GetPixelType().GetBitsPerComponent() == 16) {
        APPLY_MULTICHANNEL_RGBA(signed short);
      } else if (image->GetPixelType().GetType() == mitkIpPicUInt && image->GetPixelType().GetBitsPerComponent() == 16) {
        APPLY_MULTICHANNEL_RGBA(unsigned short);
      } else if (image->GetPixelType().GetType() == mitkIpPicFloat && image->GetPixelType().GetBitsPerComponent() == 32) {
        MITK_DEBUG << "Assuming RGB (float)\n"
            << "ITK typeID: " << typeid(itk::RGBAPixel<float>).name() << std::endl;

        APPLY_MULTICHANNEL_RGBA(float);
      } else if (image->GetPixelType().GetType()== mitkIpPicFloat && image->GetPixelType().GetBitsPerComponent() == 64) {
        APPLY_MULTICHANNEL_RGBA(double);
      } else
        MITK_ERROR << "pixel type " << image->GetPixelType().GetItkTypeAsString() << " is not supported.\n";

#undef APPLY_MULTICHANNEL_RGBA
    } else {
      AccessByItk_n(image, _ApplyTransform, (*sp_incTransform));
    }

#undef APPLY_MULTICHANNEL
  } catch (mitk::AccessByItkException &r_ex) {
    MITK_ERROR << "MITK Exception:\n" << r_ex.what() << endl;
  }
*/
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

    vtkSmartPointer<vtkMatrix4x4> identity = vtkMatrix4x4::New();
    identity->Identity();
    mitk::AffineTransformDataNodeProperty::StoreTransformInNode(COMBINED_TRANSFORM_KEY, *(identity.GetPointer()), *(msp_DataOwnerNode.GetPointer()));
    mitk::AffineTransformDataNodeProperty::StoreTransformInNode(PRELOADED_TRANSFORM_KEY, *(identity.GetPointer()), *(msp_DataOwnerNode.GetPointer()));

    // Do the resampling, according to current GUI parameters, which represent the "current" transformation.
    _ApplyResampleToCurrentNode();

    // Then reset the parameters.
    _ResetControls();
    _UpdateTransformDisplay();
  }
  QmitkAbstractView::RequestRenderWindowUpdate();
}
