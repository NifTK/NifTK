/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-08-01 16:14:30 +0100 (Mon, 01 Aug 2011) $
 Revision          : $Revision: 6909 $
 Last modified by  : $Author: ad $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <string>

// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>

// Qmitk
#include "QmitkNiftyRegView.h"

// Qt
#include <QMessageBox>
#include <QFileDialog>

// ITK
#include <itksys/SystemTools.hxx>


#include "mitkDataStorageUtils.h"
#include "mitkNodePredicateDataType.h"

#include "_reg_ReadWriteImage.h"

#include "RegistrationExecution.h"

const std::string QmitkNiftyRegView::VIEW_ID = "uk.ac.ucl.cmic.views.niftyregview";

#define USE_QT_THREADING

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

QmitkNiftyRegView::QmitkNiftyRegView()
{

  m_RegAladin = 0;
  m_RegNonRigid = 0;

  SetDefaultParameters();

}


// ---------------------------------------------------------------------------
// GetDataNode();
// --------------------------------------------------------------------------- 

mitk::DataNode::Pointer QmitkNiftyRegView::GetDataNode( QString searchName )
{
  std::string name;

  mitk::DataStorage::SetOfObjects::ConstPointer nodes = GetNodes();

  if ( nodes && (nodes->size() > 0) )
  {
    for (unsigned int i=0; i<nodes->size(); i++) 
    {
      (*nodes)[i]->GetStringProperty("name", name);
      
      if ( QString(name.c_str()) == searchName )
	return (*nodes)[i];
    }
  }

  return 0;
}


// ---------------------------------------------------------------------------
// SetDefaultParameters()
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::SetDefaultParameters()
{

  m_Modified = true;
  
  // Progress bar parameters
  m_ProgressBarOffset = 0.;
  m_ProgressBarRange = 100.;

  // Initialise the registration parameters
  m_RegParameters.SetDefaultParameters();

}


// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

QmitkNiftyRegView::~QmitkNiftyRegView()
{
  this->GetDataStorage()->AddNodeEvent
    .RemoveListener( mitk::MessageDelegate1<QmitkNiftyRegView, const mitk::DataNode*>
		     ( this, &QmitkNiftyRegView::OnNodeAdded ) );

  this->GetDataStorage()->ChangedNodeEvent
    .RemoveListener( mitk::MessageDelegate1<QmitkNiftyRegView, const mitk::DataNode*>
		     ( this, &QmitkNiftyRegView::OnNodeChanged ) );
  
  this->GetDataStorage()->RemoveNodeEvent
    .RemoveListener( mitk::MessageDelegate1<QmitkNiftyRegView, const mitk::DataNode*>
		     ( this, &QmitkNiftyRegView::OnNodeRemoved ) );

  if ( m_RegAladin ) 
    delete m_RegAladin;

  if ( m_RegNonRigid )
    delete m_RegNonRigid;
}


// ---------------------------------------------------------------------------
// SetFocus()
// ---------------------------------------------------------------------------

void QmitkNiftyRegView::SetFocus()
{
  m_Controls.m_SourceImageComboBox->setFocus();
}


// ---------------------------------------------------------------------------
// GetNodes()
// ---------------------------------------------------------------------------

mitk::DataStorage::SetOfObjects::ConstPointer QmitkNiftyRegView::GetNodes()
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();

  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage;
  isImage = mitk::TNodePredicateDataType<mitk::Image>::New();

  return dataStorage->GetSubset(isImage);
}


// ---------------------------------------------------------------------------
// CreateQtPartControl()
// ---------------------------------------------------------------------------

void QmitkNiftyRegView::CreateQtPartControl(QWidget *parent)
{
  // create GUI widgets from the Qt Designer's .ui file
  m_Controls.setupUi( parent );


  this->SetGuiToParameterValues();
  this->CreateConnections();

}


// ---------------------------------------------------------------------------
// SetGuiToParameterValues()
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::SetGuiToParameterValues()
{

  m_Controls.m_NiftyRegViewTabWidget->setCurrentIndex( 0 );

  // Initialise the source and target image combo box

  mitk::DataStorage::SetOfObjects::ConstPointer nodes = GetNodes();

  m_Controls.m_SourceImageComboBox->clear();
  m_Controls.m_TargetImageComboBox->clear();

  if ( nodes ) 
  {    
    for (unsigned int i = 0; i < nodes->size(); i++) 
    {
      std::string name;
      (*nodes)[i]->GetStringProperty("name", name);
      
      m_Controls.m_SourceImageComboBox->insertItem(i, QString(name.c_str()));
      m_Controls.m_TargetImageComboBox->insertItem(i, QString(name.c_str()));
    }
  }

  if ( ( nodes->size() > 1 ) && 
       ( m_Controls.m_SourceImageComboBox->currentIndex() == 0 ) &&
       ( m_Controls.m_TargetImageComboBox->currentIndex() == 0 ) )
  {
    m_Controls.m_SourceImageComboBox->setCurrentIndex( 1 );
  }
       
  m_Controls.m_TargetMaskImageComboBox->clear();
  m_Controls.m_TargetMaskImageComboBox->insertItem(0, QString("no mask"));


  // Multi-Scale Options
    
  m_Controls.m_NumberOfLevelsSpinBox->setKeyboardTracking ( false );
  m_Controls.m_NumberOfLevelsSpinBox->setMaximum( 10 );
  m_Controls.m_NumberOfLevelsSpinBox->setMinimum( 1 );
  m_Controls.m_NumberOfLevelsSpinBox->setValue( m_RegParameters.m_LevelNumber );

  m_Controls.m_LevelsToPerformSpinBox->setKeyboardTracking ( false );
  m_Controls.m_LevelsToPerformSpinBox->setMaximum( 10 );
  m_Controls.m_LevelsToPerformSpinBox->setMinimum( 1 );
  m_Controls.m_LevelsToPerformSpinBox->setValue( m_RegParameters.m_Level2Perform );

  // Input Image Options
 
  m_Controls.m_SmoothSourceImageDoubleSpinBox->setKeyboardTracking ( false );
  m_Controls.m_SmoothSourceImageDoubleSpinBox->setMaximum( 1000 );
  m_Controls.m_SmoothSourceImageDoubleSpinBox->setMinimum( 0 );
  m_Controls.m_SmoothSourceImageDoubleSpinBox->setValue( m_RegParameters.m_SourceSigmaValue );

  m_Controls.m_SmoothTargetImageDoubleSpinBox->setKeyboardTracking ( false );
  m_Controls.m_SmoothTargetImageDoubleSpinBox->setMaximum( 1000 );
  m_Controls.m_SmoothTargetImageDoubleSpinBox->setMinimum( 0 );
  m_Controls.m_SmoothTargetImageDoubleSpinBox->setValue( m_RegParameters.m_TargetSigmaValue );

  m_Controls.m_DoBlockMatchingOnlyRadioButton->setChecked( m_RegParameters.m_FlagDoInitialRigidReg 
							   && ( ! m_RegParameters.m_FlagDoNonRigidReg ) );
  m_Controls.m_DoNonRigidOnlyRadioButton->setChecked( ( ! m_RegParameters.m_FlagDoInitialRigidReg )
						      && m_RegParameters.m_FlagDoNonRigidReg );
  m_Controls.m_DoBlockMatchingThenNonRigidRadioButton->setChecked( m_RegParameters.m_FlagDoInitialRigidReg 
								   && m_RegParameters.m_FlagDoNonRigidReg );


  // Initial Affine Transformation

  m_Controls.m_InitialAffineTransformationGroupBox->setChecked( m_RegParameters.m_FlagInputAffine );
  m_Controls.m_InputFlirtCheckBox->setChecked( m_RegParameters.m_FlagFlirtAffine );

  if ( ! m_RegParameters.m_InputAffineName.isEmpty() )
    m_Controls.m_InputAffineFileNameLineEdit->setText( m_RegParameters.m_InputAffineName );
  else
    m_Controls.m_InputAffineFileNameLineEdit->setText( QString( "" ) );


  // Aladin Parameters
  // ~~~~~~~~~~~~~~~~~

  // Aladin - Initialisation

  m_Controls.m_UseNiftyHeaderCheckBox->setChecked( m_RegParameters.m_AladinParameters.alignCenterFlag );

  // Aladin - Method

  m_Controls.m_RigidOnlyRadioButton->setChecked( m_RegParameters.m_AladinParameters.regnType 
						 == RIGID_ONLY );
  m_Controls.m_RigidThenAffineRadioButton->setChecked( m_RegParameters.m_AladinParameters.regnType 
						       == RIGID_THEN_AFFINE );
  m_Controls.m_DirectAffineRadioButton->setChecked( m_RegParameters.m_AladinParameters.regnType 
						    == DIRECT_AFFINE );

  m_Controls.m_AladinUseSymmetricAlgorithmCheckBox->setChecked( m_RegParameters.m_AladinParameters.symFlag );

  m_Controls.m_AladinIterationsMaxSpinBox->setKeyboardTracking ( false );
  m_Controls.m_AladinIterationsMaxSpinBox->setMaximum( 1000 );
  m_Controls.m_AladinIterationsMaxSpinBox->setMinimum( 1 );
  m_Controls.m_AladinIterationsMaxSpinBox->setValue( m_RegParameters.m_AladinParameters.maxiterationNumber );

  m_Controls.m_PercentBlockSpinBox->setKeyboardTracking ( false );
  m_Controls.m_PercentBlockSpinBox->setMaximum( 100 );
  m_Controls.m_PercentBlockSpinBox->setMinimum( 1 );
  m_Controls.m_PercentBlockSpinBox->setValue( m_RegParameters.m_AladinParameters.block_percent_to_use );

  m_Controls.m_PercentInliersSpinBox->setKeyboardTracking ( false );
  m_Controls.m_PercentInliersSpinBox->setMaximum( 100 );
  m_Controls.m_PercentInliersSpinBox->setMinimum( 1 );
  m_Controls.m_PercentInliersSpinBox->setValue( m_RegParameters.m_AladinParameters.inlier_lts );

  // Aladin - Advanced

  m_Controls.m_AladinInterpolationNearestRadioButton->setChecked( m_RegParameters.m_AladinParameters
								  .interpolation 
								  == NEAREST_INTERPOLATION );
  m_Controls.m_AladinInterpolationLinearRadioButton->setChecked( m_RegParameters.m_AladinParameters
								  .interpolation 
								  == LINEAR_INTERPOLATION );
  m_Controls.m_AladinInterpolationCubicRadioButton->setChecked( m_RegParameters.m_AladinParameters
								.interpolation 
								== CUBIC_INTERPOLATION );

  // Non-Rigid Parameters
  // ~~~~~~~~~~~~~~~~~~~~
  
  // Non-Rigid - Initialisation

  m_Controls.m_NonRigidInputControlPointCheckBox->setChecked( m_RegParameters.m_F3dParameters.
							      inputControlPointGridFlag );

  if ( ! m_RegParameters.m_F3dParameters.inputControlPointGridName.isEmpty() )
    m_Controls.m_NonRigidInputControlPointFileNameLineEdit->setText( m_RegParameters.
								     m_F3dParameters.
								     inputControlPointGridName );
  else
    m_Controls.m_NonRigidInputControlPointFileNameLineEdit->setText( QString( "" ) );

  m_Controls.m_NonRigidInputControlPointFileNameLineEdit->setEnabled( m_RegParameters.
								      m_F3dParameters.
								      inputControlPointGridFlag );

  // Non-Rigid - Input Image

  // LowerThresholdTargetImageCheckBox
  m_Controls.m_LowerThresholdTargetImageDoubleSpinBox->setKeyboardTracking ( false );

  m_Controls.m_LowerThresholdTargetImageDoubleSpinBox
    ->setMaximum( std::numeric_limits<PrecisionTYPE>::max() );
  m_Controls.m_LowerThresholdTargetImageDoubleSpinBox
    ->setMinimum( -std::numeric_limits<PrecisionTYPE>::max() );

  m_Controls.m_LowerThresholdTargetImageDoubleSpinBox->setSpecialValueText(tr("min"));

  m_Controls.m_LowerThresholdTargetImageDoubleSpinBox
    ->setValue( m_RegParameters.m_F3dParameters.referenceThresholdLow );

  // UpperThresholdTargetImageCheckBox
  m_Controls.m_UpperThresholdTargetImageDoubleSpinBox->setKeyboardTracking ( false );

  m_Controls.m_UpperThresholdTargetImageDoubleSpinBox
    ->setMaximum( std::numeric_limits<PrecisionTYPE>::max() );
  m_Controls.m_UpperThresholdTargetImageDoubleSpinBox
    ->setMinimum( -std::numeric_limits<PrecisionTYPE>::max() );

  m_Controls.m_UpperThresholdTargetImageDoubleSpinBox->setSpecialValueText(tr("max"));

  m_Controls.m_UpperThresholdTargetImageDoubleSpinBox
    ->setValue( m_RegParameters.m_F3dParameters.referenceThresholdUp );

  // LowerThresholdSourceImage
  m_Controls.m_LowerThresholdSourceImageDoubleSpinBox->setKeyboardTracking ( false );

  m_Controls.m_LowerThresholdSourceImageDoubleSpinBox
    ->setMaximum( std::numeric_limits<PrecisionTYPE>::max() );
  m_Controls.m_LowerThresholdSourceImageDoubleSpinBox
    ->setMinimum( -std::numeric_limits<PrecisionTYPE>::max() );

  m_Controls.m_LowerThresholdSourceImageDoubleSpinBox->setSpecialValueText(tr("min"));

  m_Controls.m_LowerThresholdSourceImageDoubleSpinBox
    ->setValue( m_RegParameters.m_F3dParameters.floatingThresholdLow );

  // UpperThresholdSourceImage
  m_Controls.m_UpperThresholdSourceImageDoubleSpinBox->setKeyboardTracking ( false );

  m_Controls.m_UpperThresholdSourceImageDoubleSpinBox
    ->setMaximum( std::numeric_limits<PrecisionTYPE>::max() );
  m_Controls.m_UpperThresholdSourceImageDoubleSpinBox
    ->setMinimum( -std::numeric_limits<PrecisionTYPE>::max() );

  m_Controls.m_UpperThresholdSourceImageDoubleSpinBox->setSpecialValueText(tr("max"));

  m_Controls.m_UpperThresholdSourceImageDoubleSpinBox
    ->setValue( m_RegParameters.m_F3dParameters.floatingThresholdUp );

  // Non-Rigid - Spline

  m_Controls.m_ControlPointSpacingXDoubleSpinBox->setMinimum( -50 );
  m_Controls.m_ControlPointSpacingXDoubleSpinBox->setMaximum(  50 );
  m_Controls.m_ControlPointSpacingXDoubleSpinBox->setValue( m_RegParameters.m_F3dParameters.spacing[0] );

  m_Controls.m_ControlPointSpacingYDoubleSpinBox->setMinimum( -50 );
  m_Controls.m_ControlPointSpacingYDoubleSpinBox->setMaximum(  50 );
  m_Controls.m_ControlPointSpacingYDoubleSpinBox->setValue( m_RegParameters.m_F3dParameters.spacing[1] );

  m_Controls.m_ControlPointSpacingZDoubleSpinBox->setMinimum( -50 );
  m_Controls.m_ControlPointSpacingZDoubleSpinBox->setMaximum(  50 );
  m_Controls.m_ControlPointSpacingZDoubleSpinBox->setValue( m_RegParameters.m_F3dParameters.spacing[2] );

  // Non-Rigid - Objective Function
    
  m_Controls.m_NumberSourceHistogramBinsSpinBox->setMinimum( 4 );
  m_Controls.m_NumberSourceHistogramBinsSpinBox->setMaximum( 256 );
  m_Controls.m_NumberSourceHistogramBinsSpinBox->setValue( m_RegParameters.m_F3dParameters.
							   referenceBinNumber );

  m_Controls.m_NumberTargetHistogramBinsSpinBox->setMinimum( 4 );
  m_Controls.m_NumberTargetHistogramBinsSpinBox->setMaximum( 256 );
  m_Controls.m_NumberTargetHistogramBinsSpinBox->setValue( m_RegParameters.m_F3dParameters.
							   floatingBinNumber );

  m_Controls.m_WeightBendingEnergyDoubleSpinBox->setMinimum( 0 );
  m_Controls.m_WeightBendingEnergyDoubleSpinBox->setMaximum( 1 );
  m_Controls.m_WeightBendingEnergyDoubleSpinBox->setValue( m_RegParameters.m_F3dParameters.
							   bendingEnergyWeight );

  m_Controls.m_WeightLogJacobianDoubleSpinBox->setMinimum( 0 );
  m_Controls.m_WeightLogJacobianDoubleSpinBox->setMaximum( 1 );
  m_Controls.m_WeightLogJacobianDoubleSpinBox->setValue( m_RegParameters.m_F3dParameters.
							 jacobianLogWeight );

  m_Controls.m_LinearEnergyWeightsDoubleSpinBox_1->setMinimum( 0 );
  m_Controls.m_LinearEnergyWeightsDoubleSpinBox_1->setMaximum( 100 );
  m_Controls.m_LinearEnergyWeightsDoubleSpinBox_1->setValue( m_RegParameters.m_F3dParameters.
							     linearEnergyWeight0 );

  m_Controls.m_LinearEnergyWeightsDoubleSpinBox_2->setMinimum( 0 );
  m_Controls.m_LinearEnergyWeightsDoubleSpinBox_2->setMaximum( 100 );
  m_Controls.m_LinearEnergyWeightsDoubleSpinBox_2->setValue( m_RegParameters.m_F3dParameters.
							     linearEnergyWeight1 );

  m_Controls.m_ApproxJacobianLogCheckBox->setChecked( m_RegParameters.m_F3dParameters.
						      jacobianLogApproximation );

  m_Controls.m_SimilarityNMIRadioButton->setChecked( m_RegParameters.m_F3dParameters.similarity 
						     == NMI_SIMILARITY );
  m_Controls.m_SimilaritySSDRadioButton->setChecked( m_RegParameters.m_F3dParameters.similarity 
						     == SSD_SIMILARITY );
  m_Controls.m_SimilarityKLDivRadioButton->setChecked( m_RegParameters.m_F3dParameters.similarity 
						       == KLDIV_SIMILARITY );

  // Non-Rigid - Optimisation

  m_Controls.m_UseSimpleGradientAscentCheckBox->setChecked( ! m_RegParameters.m_F3dParameters.useConjugate );

  m_Controls.m_NonRigidIterationsMaxSpinBox->setMaximum( 100 );
  m_Controls.m_NonRigidIterationsMaxSpinBox->setMinimum( 1 );
  m_Controls.m_NonRigidIterationsMaxSpinBox->setValue( m_RegParameters.m_F3dParameters.maxiterationNumber );

  m_Controls.m_UsePyramidalCheckBox->setChecked( ! m_RegParameters.m_F3dParameters.noPyramid );

  // Non-Rigid - Advanced

  m_Controls.m_SmoothingMetricDoubleSpinBox->setMaximum( 50 );
  m_Controls.m_SmoothingMetricDoubleSpinBox->setMinimum( 0 );
  m_Controls.m_SmoothingMetricDoubleSpinBox->setValue( m_RegParameters.m_F3dParameters.gradientSmoothingSigma );

  m_Controls.m_WarpedPaddingValueDoubleSpinBox
    ->setMaximum( std::numeric_limits<PrecisionTYPE>::max() );
  m_Controls.m_WarpedPaddingValueDoubleSpinBox
    ->setMinimum( -std::numeric_limits<PrecisionTYPE>::max() );
  m_Controls.m_WarpedPaddingValueDoubleSpinBox->setSpecialValueText(tr("none"));
  m_Controls.m_WarpedPaddingValueDoubleSpinBox->setValue( m_RegParameters.m_F3dParameters.warpedPaddingValue );


  m_Controls.m_NonRigidNearestInterpolationRadioButton->setChecked( m_RegParameters.m_F3dParameters.
								    interpolation 
								   == NEAREST_INTERPOLATION );

  m_Controls.m_NonRigidLinearInterpolationRadioButton->setChecked( m_RegParameters.m_F3dParameters.
								   interpolation 
								   == LINEAR_INTERPOLATION );

  m_Controls.m_NonRigidCubicInterpolationRadioButton->setChecked( m_RegParameters.m_F3dParameters.
								  interpolation 
								   == CUBIC_INTERPOLATION );
}


// ---------------------------------------------------------------------------
// CreateConnections()
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::CreateConnections()
{

  // Register data storage listeners

  this->GetDataStorage()->AddNodeEvent
    .AddListener( mitk::MessageDelegate1<QmitkNiftyRegView, const mitk::DataNode*>
		  ( this, &QmitkNiftyRegView::OnNodeAdded ) );

  this->GetDataStorage()->ChangedNodeEvent
    .AddListener( mitk::MessageDelegate1<QmitkNiftyRegView, const mitk::DataNode*>
		  ( this, &QmitkNiftyRegView::OnNodeChanged ) );

  this->GetDataStorage()->RemoveNodeEvent
    .AddListener( mitk::MessageDelegate1<QmitkNiftyRegView, const mitk::DataNode*>
		  ( this, &QmitkNiftyRegView::OnNodeRemoved ) );

  // Input Image Options

  connect( m_Controls.m_SourceImageComboBox,
	   SIGNAL( currentIndexChanged( int ) ),
	   this,
	   SLOT( OnSourceImageComboBoxChanged( int ) ) );

  connect( m_Controls.m_TargetImageComboBox,
	   SIGNAL( currentIndexChanged( int ) ),
	   this,
	   SLOT( OnTargetImageComboBoxChanged( int ) ) );

  connect( m_Controls.m_TargetMaskImageComboBox,
	   SIGNAL( currentIndexChanged( int ) ),
	   this,
	   SLOT( OnTargetMaskImageComboBoxChanged( int ) ) );


  // Multi-Scale Options
    
  connect( m_Controls.m_NumberOfLevelsSpinBox,
	   SIGNAL( valueChanged( int ) ),
	   this,
	   SLOT( OnNumberOfLevelsSpinBoxValueChanged( int ) ) );

  connect( m_Controls.m_LevelsToPerformSpinBox,
	   SIGNAL( valueChanged( int ) ),
	   this,
	   SLOT( OnLevelsToPerformSpinBoxValueChanged( int ) ) );

  // Input Image Options

  connect( m_Controls.m_SmoothSourceImageDoubleSpinBox,
	   SIGNAL( valueChanged( double ) ),
	   this,
	   SLOT( OnSmoothSourceImageDoubleSpinBoxValueChanged( double ) ) );

  connect( m_Controls.m_SmoothTargetImageDoubleSpinBox,
	   SIGNAL( valueChanged( double ) ),
	   this,
	   SLOT( OnSmoothTargetImageDoubleSpinBoxValueChanged( double ) ) );

  connect( m_Controls.m_NoSmoothingPushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnNoSmoothingPushButtonPressed( void ) ) );


  // Flag indicating whether to do an initial rigid registration etc.

  connect( m_Controls.m_DoBlockMatchingOnlyRadioButton,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnDoBlockMatchingOnlyRadioButtonToggled( bool ) ) );

  connect( m_Controls.m_DoNonRigidOnlyRadioButton,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnDoNonRigidOnlyRadioButtonToggled( bool ) ) );

  connect( m_Controls.m_DoBlockMatchingThenNonRigidRadioButton,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnDoBlockMatchingThenNonRigidRadioButtonToggled( bool ) ) );


  // Initialise the 'reg_aladin' parameters
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Aladin - Initialisation

  connect( m_Controls.m_InitialAffineTransformationGroupBox,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnInputAffineCheckBoxToggled( bool ) ) );

  connect( m_Controls.m_InputAffineBrowsePushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnInputAffineBrowsePushButtonPressed( void ) ) );

  connect( m_Controls.m_InputFlirtCheckBox,
	   SIGNAL( stateChanged( int ) ),
	   this,
	   SLOT( OnInputFlirtCheckBoxStateChanged( int ) ) );

  connect( m_Controls.m_UseNiftyHeaderCheckBox,
	   SIGNAL( stateChanged( int ) ),
	   this,
	   SLOT( OnUseNiftyHeaderCheckBoxStateChanged( int ) ) );
  
  // Aladin - Method

  connect( m_Controls.m_RigidOnlyRadioButton,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnRigidOnlyRadioButtonToggled( bool ) ) );

  connect( m_Controls.m_RigidThenAffineRadioButton,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnRigidThenAffineRadioButtonToggled( bool ) ) );

  connect( m_Controls.m_DirectAffineRadioButton,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnDirectAffineRadioButtonToggled( bool ) ) );


  connect( m_Controls.m_AladinUseSymmetricAlgorithmCheckBox,
	   SIGNAL( stateChanged( int ) ),
	   this,
	   SLOT( OnAladinUseSymmetricAlgorithmCheckBoxStateChanged( int ) ) );


  connect( m_Controls.m_AladinIterationsMaxSpinBox,
	   SIGNAL( valueChanged( int ) ),
	   this,
	   SLOT( OnAladinIterationsMaxSpinBoxValueChanged( int ) ) );

  connect( m_Controls.m_PercentBlockSpinBox,
	   SIGNAL( valueChanged( int ) ),
	   this,
	   SLOT( OnPercentBlockSpinBoxValueChanged( int ) ) );

  connect( m_Controls.m_PercentInliersSpinBox,
	   SIGNAL( valueChanged( int ) ),
	   this,
	   SLOT( OnPercentInliersSpinBoxValueChanged( int ) ) );

  // Aladin - Advanced

  connect( m_Controls.m_AladinInterpolationNearestRadioButton,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnAladinInterpolationNearestRadioButtonToggled( bool ) ) );

  connect( m_Controls.m_AladinInterpolationLinearRadioButton,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnAladinInterpolationLinearRadioButtonToggled( bool ) ) );

  connect( m_Controls.m_AladinInterpolationCubicRadioButton,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnAladinInterpolationCubicRadioButtonToggled( bool ) ) );


  // Initialise the 'reg_f3d' parameters
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Non-Rigid - Initialisation
 
  connect( m_Controls.m_NonRigidInputControlPointCheckBox,
	   SIGNAL( stateChanged( int ) ),
	   this,
	   SLOT( OnNonRigidInputControlPointCheckBoxStateChanged( int ) ) );

  connect( m_Controls.m_NonRigidInputControlPointBrowsePushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnNonRigidInputControlPointBrowsePushButtonPressed( void ) ) );

  // Non-Rigid - Input Image

  connect( m_Controls.m_LowerThresholdTargetImageDoubleSpinBox,
	   SIGNAL( valueChanged( double ) ),
	   this,
	   SLOT( OnLowerThresholdTargetImageDoubleSpinBoxValueChanged( double ) ) );

  connect( m_Controls.m_UpperThresholdTargetImageDoubleSpinBox,
	   SIGNAL( valueChanged( double ) ),
	   this,
	   SLOT( OnUpperThresholdTargetImageDoubleSpinBoxValueChanged( double ) ) );

  connect( m_Controls.m_UpperThresholdTargetImageAutoPushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnUpperThresholdTargetImageAutoPushButtonPressed( void ) ) );

  connect( m_Controls.m_LowerThresholdTargetImageAutoPushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnLowerThresholdTargetImageAutoPushButtonPressed( void ) ) );

  connect( m_Controls.m_LowerThresholdSourceImageDoubleSpinBox,
	   SIGNAL( valueChanged( double ) ),
	   this,
	   SLOT( OnLowerThresholdSourceImageDoubleSpinBoxValueChanged( double ) ) );

  connect( m_Controls.m_UpperThresholdSourceImageDoubleSpinBox,
	   SIGNAL( valueChanged( double ) ),
	   this,
	   SLOT( OnUpperThresholdSourceImageDoubleSpinBoxValueChanged( double ) ) );

  connect( m_Controls.m_UpperThresholdSourceImageAutoPushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnUpperThresholdSourceImageAutoPushButtonPressed( void ) ) );

  connect( m_Controls.m_LowerThresholdSourceImageAutoPushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnLowerThresholdSourceImageAutoPushButtonPressed( void ) ) );

  // Non-Rigid - Spline

  connect( m_Controls.m_ControlPointSpacingXDoubleSpinBox,
	   SIGNAL( valueChanged( double ) ),
	   this,
	   SLOT( OnControlPointSpacingXDoubleSpinBoxValueChanged( double ) ) );

  connect( m_Controls.m_ControlPointSpacingYDoubleSpinBox,
	   SIGNAL( valueChanged( double ) ),
	   this,
	   SLOT( OnControlPointSpacingYDoubleSpinBoxValueChanged( double ) ) );

  connect( m_Controls.m_ControlPointSpacingZDoubleSpinBox,
	   SIGNAL( valueChanged( double ) ),
	   this,
	   SLOT( OnControlPointSpacingZDoubleSpinBoxValueChanged( double ) ) );

  // Non-Rigid - Objective Function
 
  connect( m_Controls.m_NumberSourceHistogramBinsSpinBox,
	   SIGNAL( valueChanged( int ) ),
	   this,
	   SLOT( OnNumberSourceHistogramBinsSpinBoxValueChanged( int ) ) );

  connect( m_Controls.m_NumberTargetHistogramBinsSpinBox,
	   SIGNAL( valueChanged( int ) ),
	   this,
	   SLOT( OnNumberTargetHistogramBinsSpinBoxValueChanged( int ) ) );

  connect( m_Controls.m_WeightBendingEnergyDoubleSpinBox,
	   SIGNAL( valueChanged( double ) ),
	   this,
	   SLOT( OnWeightBendingEnergyDoubleSpinBoxValueChanged( double ) ) );

  connect( m_Controls.m_WeightLogJacobianDoubleSpinBox,
	   SIGNAL( valueChanged( double ) ),
	   this,
	   SLOT( OnWeightLogJacobianDoubleSpinBoxValueChanged( double ) ) );

  connect( m_Controls.m_LinearEnergyWeightsDoubleSpinBox_1,
	   SIGNAL( valueChanged( double ) ),
	   this,
	   SLOT( OnLinearEnergyWeightsDoubleSpinBox_1ValueChanged( double ) ) );

  connect( m_Controls.m_LinearEnergyWeightsDoubleSpinBox_2,
	   SIGNAL( valueChanged( double ) ),
	   this,
	   SLOT( OnLinearEnergyWeightsDoubleSpinBox_2ValueChanged( double ) ) );

  connect( m_Controls.m_ApproxJacobianLogCheckBox,
	   SIGNAL( stateChanged( int ) ),
	   this,
	   SLOT( OnApproxJacobianLogCheckBoxStateChanged( int ) ) );

  connect( m_Controls.m_SimilarityNMIRadioButton,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnSimilarityNMIRadioButtonToggled( bool ) ) );

  connect( m_Controls.m_SimilaritySSDRadioButton,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnSimilaritySSDRadioButtonToggled( bool ) ) );

  connect( m_Controls.m_SimilarityKLDivRadioButton,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnSimilarityKLDivRadioButtonToggled( bool ) ) );

  // Non-Rigid - Optimisation
 
  connect( m_Controls.m_UseSimpleGradientAscentCheckBox,
	   SIGNAL( stateChanged( int ) ),
	   this,
	   SLOT( OnUseSimpleGradientAscentCheckBoxStateChanged( int ) ) );

  connect( m_Controls.m_UsePyramidalCheckBox,
	   SIGNAL( stateChanged( int ) ),
	   this,
	   SLOT( OnUsePyramidalCheckBoxStateChanged( int ) ) );

  connect( m_Controls.m_NonRigidIterationsMaxSpinBox,
	   SIGNAL( valueChanged( int ) ),
	   this,
	   SLOT( OnNonRigidIterationsMaxSpinBoxValueChanged( int ) ) );

  // Non-Rigid - Advanced

  connect( m_Controls.m_SmoothingMetricDoubleSpinBox,
	   SIGNAL( valueChanged( double ) ),
	   this,
	   SLOT( OnSmoothingMetricDoubleSpinBoxValueChanged( double ) ) );

  connect( m_Controls.m_WarpedPaddingValueDoubleSpinBox,
	   SIGNAL( valueChanged( double ) ),
	   this,
	   SLOT( OnWarpedPaddingValueDoubleSpinBoxValueChanged( double ) ) );

  connect( m_Controls.m_WarpedPaddingValuePushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnWarpedPaddingValuePushButtonPressed( void ) ) );


  connect( m_Controls.m_NonRigidNearestInterpolationRadioButton,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnNonRigidNearestInterpolationRadioButtonToggled( bool ) ) );

  connect( m_Controls.m_NonRigidLinearInterpolationRadioButton,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnNonRigidLinearInterpolationRadioButtonToggled( bool ) ) );

  connect( m_Controls.m_NonRigidCubicInterpolationRadioButton,
	   SIGNAL( toggled( bool ) ),
	   this,
	   SLOT( OnNonRigidCubicInterpolationRadioButtonToggled( bool ) ) );


  // Execution etc.

#if 0
  connect( m_Controls.m_CancelPushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnCancelPushButtonPressed( void ) ) );
#endif

  connect( m_Controls.m_ResetParametersPushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnResetParametersPushButtonPressed( void ) ) );

  connect( m_Controls.m_SaveTransformationPushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnSaveTransformationPushButtonPressed( void ) ) );

  connect( m_Controls.m_ExecutePushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnExecutePushButtonPressed( void ) ) );


  connect( m_Controls.m_SaveRegistrationParametersPushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnSaveRegistrationParametersPushButtonPressed( void ) ) );

  connect( m_Controls.m_LoadRegistrationParametersPushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnLoadRegistrationParametersPushButtonPressed( void ) ) );

}


// ---------------------------------------------------------------------------
// PrintSelf
// ---------------------------------------------------------------------------

void QmitkNiftyRegView::PrintSelf( std::ostream& os )
{

  m_RegParameters.PrintSelf( os );

}


// ---------------------------------------------------------------------------
// Modified
// ---------------------------------------------------------------------------

void QmitkNiftyRegView::Modified()
{
  m_Controls.m_ExecutePushButton->setEnabled( true );

  m_Controls.m_ProgressBar->reset();

  m_Modified = true;
}


// ---------------------------------------------------------------------------
// OnNodeAdded()
// ---------------------------------------------------------------------------

void QmitkNiftyRegView::OnNodeAdded(const mitk::DataNode* node)
{
  std::string name;
  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage;

  node->GetStringProperty("name", name);

  isImage = mitk::TNodePredicateDataType<mitk::Image>::New();

  if ( isImage->CheckNode( node ) ) 
  {
    m_Controls.m_SourceImageComboBox->addItem( QString(name.c_str()) );
    m_Controls.m_TargetImageComboBox->addItem( QString(name.c_str()) );
    m_Controls.m_TargetMaskImageComboBox->addItem( QString(name.c_str()) );
  }

  mitk::DataStorage::SetOfObjects::ConstPointer nodes = GetNodes();

  if ( ( nodes ) && ( nodes->size() == 2 ) )
  {
    m_Controls.m_SourceImageComboBox->setCurrentIndex( 1 );
    m_Controls.m_TargetImageComboBox->setCurrentIndex( 0 );
  }

}


// ---------------------------------------------------------------------------
// OnNodeRemoved()
// ---------------------------------------------------------------------------

void QmitkNiftyRegView::OnNodeRemoved(const mitk::DataNode* node)
{
  int index;
  std::string name;

  node->GetStringProperty("name", name);

  // SourceImageComboBox
  index = m_Controls.m_SourceImageComboBox->findText( QString(name.c_str()) );

  if ( index >= 0 )
  {
    m_Controls.m_SourceImageComboBox->removeItem( index );
  } 

  // TargetImageComboBox
  index = m_Controls.m_TargetImageComboBox->findText( QString(name.c_str()) );
  
  if ( index >= 0 )
  {
    m_Controls.m_TargetImageComboBox->removeItem( index );
  } 

  // TargetMaskImageComboBox
  index = m_Controls.m_TargetMaskImageComboBox->findText( QString(name.c_str()) );
  
  if ( index >= 0 )
  {
    m_Controls.m_TargetMaskImageComboBox->removeItem( index );
  } 

}


// ---------------------------------------------------------------------------
// OnNodeChanged()
// ---------------------------------------------------------------------------

void QmitkNiftyRegView::OnNodeChanged(const mitk::DataNode* node)
{
  int index;
  std::string name;

  node->GetStringProperty("name", name);

  // SourceImageComboBox
  index = m_Controls.m_SourceImageComboBox->findText( QString(name.c_str()) );
  
  if ( (index != -1) && (index == m_Controls.m_SourceImageComboBox->currentIndex()) )
  {
    Modified();
  }

  // TargetImageComboBox
  index = m_Controls.m_TargetImageComboBox->findText( QString(name.c_str()) );
  
  if ( (index != -1) && (index == m_Controls.m_TargetImageComboBox->currentIndex()) )
  {
    Modified();
  }

  // TargetMaskImageComboBox
  index = m_Controls.m_TargetMaskImageComboBox->findText( QString(name.c_str()) );
  
  if ( (index != -1) && (index == m_Controls.m_TargetMaskImageComboBox->currentIndex()) )
  {
    Modified();
  }

}


// ---------------------------------------------------------------------------
// OnTargetImageComboBoxChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnTargetImageComboBoxChanged(int index)
{
  std::string filepath, filename;

  QString sourceName = m_Controls.m_SourceImageComboBox->currentText();
  QString targetName = m_Controls.m_TargetImageComboBox->currentText();

  if ( ! targetName.isEmpty() ) 
  {
    mitk::DataNode::Pointer node = GetDataNode( targetName );

    if ( node ) 
    {
      node->GetStringProperty( mitk::StringProperty::PATH, filepath );
      node->GetStringProperty( "name", filename );

      m_RegParameters.m_AladinParameters.referenceImageName = QString( filename.c_str() );
      m_RegParameters.m_AladinParameters.referenceImagePath = QString( filepath.c_str() );

      m_RegParameters.m_F3dParameters.referenceImageName = QString( filename.c_str() );
      m_RegParameters.m_F3dParameters.referenceImagePath = QString( filepath.c_str() );
    }
  }

  if ( ( ! sourceName.isEmpty() ) &&
       ( ! targetName.isEmpty() ) )
  {
    m_Controls.m_ExecutePushButton->setEnabled( true );
  }
  else
  {
    m_Controls.m_ExecutePushButton->setEnabled( false );
  }

  Modified();
}


// ---------------------------------------------------------------------------
// UpdateAladinResultImageFilename();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::UpdateAladinResultImageFilename()
{
  std::string filepath, filename;

  QString sourceName = m_Controls.m_SourceImageComboBox->currentText();
  QString targetName = m_Controls.m_TargetImageComboBox->currentText();

  if ( ! sourceName.isEmpty() ) 
  {
    mitk::DataNode::Pointer node = GetDataNode( sourceName );

    if ( node ) 
    {
      node->GetStringProperty( mitk::StringProperty::PATH, filepath );
      node->GetStringProperty( "name", filename );

      std::string nameOfResultImage = filename;

      if ( m_RegParameters.m_AladinParameters.regnType == RIGID_ONLY )
	nameOfResultImage.append( "_RigidRegnTo_" );
      else
	nameOfResultImage.append( "_AffineRegnTo_" );
      
      nameOfResultImage.append( targetName.toStdString() );

      m_RegParameters.m_AladinParameters.outputResultName = QString( nameOfResultImage.c_str() );
      m_RegParameters.m_AladinParameters.outputResultPath = QString( filepath.c_str() );

      m_RegParameters.m_AladinParameters.outputResultFlag = true;

      // The input to the non-rigid registration might be the output of
      // the affine regn, so we need to update the non-rigid floating
      // image name also.
      
      if ( m_RegParameters.m_FlagDoInitialRigidReg ) 
      {
	m_RegParameters.m_F3dParameters.floatingImageName = QString( nameOfResultImage.c_str() );
	m_RegParameters.m_F3dParameters.floatingImagePath = QString( filepath.c_str() );
      }
      else {
	m_RegParameters.m_F3dParameters.floatingImageName = QString( filename.c_str() );
	m_RegParameters.m_F3dParameters.floatingImagePath = QString( filepath.c_str() );
      } 
    }
  }
}


// ---------------------------------------------------------------------------
// UpdateNonRigidResultImageFilename();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::UpdateNonRigidResultImageFilename()
{
  std::string filepath, filename;

  QString sourceName = m_Controls.m_SourceImageComboBox->currentText();
  QString targetName = m_Controls.m_TargetImageComboBox->currentText();

  if ( ! sourceName.isEmpty() ) 
  {
    mitk::DataNode::Pointer node = GetDataNode( sourceName );

    if ( node ) 
    {
      node->GetStringProperty( mitk::StringProperty::PATH, filepath );
      node->GetStringProperty( "name", filename );
      
      std::string nameOfResultImage = filename;
      
      if ( m_RegParameters.m_FlagDoInitialRigidReg ) 
      {
	if ( m_RegParameters.m_AladinParameters.regnType == RIGID_ONLY )
	  nameOfResultImage.append( "_RigidAndNonRigidRegnTo_" );
	else
	  nameOfResultImage.append( "_AffineAndNonRigidRegnTo_" );
      }
      else
	nameOfResultImage.append( "_NonRigidRegnTo_" );

      nameOfResultImage.append( targetName.toStdString() );

      m_RegParameters.m_F3dParameters.outputWarpedName = QString( nameOfResultImage.c_str() );
      m_RegParameters.m_F3dParameters.outputWarpedPath = QString( filepath.c_str() );
    }
  }
}


// ---------------------------------------------------------------------------
// OnSourceImageComboBoxChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnSourceImageComboBoxChanged(int index)
{
  std::string filepath, filename;

  QString sourceName = m_Controls.m_SourceImageComboBox->currentText();
  QString targetName = m_Controls.m_TargetImageComboBox->currentText();

  if ( ! sourceName.isEmpty() ) 
  {
    mitk::DataNode::Pointer node = GetDataNode( sourceName );

    if ( node ) 
    {
      node->GetStringProperty( mitk::StringProperty::PATH, filepath );
      node->GetStringProperty( "name", filename );

      m_RegParameters.m_AladinParameters.floatingImageName = QString( filename.c_str() );
      m_RegParameters.m_AladinParameters.floatingImagePath = QString( filepath.c_str() );

      UpdateAladinResultImageFilename();
      UpdateNonRigidResultImageFilename();
    }
  }

  if ( ( ! sourceName.isEmpty() ) &&
       ( ! targetName.isEmpty() ) )
  {
    m_Controls.m_ExecutePushButton->setEnabled( true );
  }
  else
  {
    m_Controls.m_ExecutePushButton->setEnabled( false );
  }

  Modified();
}


// ---------------------------------------------------------------------------
// OnTargetMaskImageComboBoxChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnTargetMaskImageComboBoxChanged(int index)
{
  std::string filepath, filename;

  QString maskName = m_Controls.m_TargetMaskImageComboBox->currentText();

  if ( ! maskName.isEmpty() ) 
  {
    mitk::DataNode::Pointer node = GetDataNode( maskName );

    if ( node ) 
    {
      node->GetStringProperty( mitk::StringProperty::PATH, filepath );
      node->GetStringProperty( "name", filename );

      m_RegParameters.m_AladinParameters.referenceMaskName = QString( filename.c_str() );
      m_RegParameters.m_AladinParameters.referenceMaskPath = QString( filepath.c_str() );

      m_RegParameters.m_F3dParameters.referenceMaskName = QString( filename.c_str() );
      m_RegParameters.m_F3dParameters.referenceMaskPath = QString( filepath.c_str() );

      Modified();
    }
  }
}


// ---------------------------------------------------------------------------
// OnCancelPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnCancelPushButtonPressed( void )
{

}


// ---------------------------------------------------------------------------
// OnResetParametersPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnResetParametersPushButtonPressed( void )
{
  SetDefaultParameters();
  SetGuiToParameterValues();

  Modified();
}


// ---------------------------------------------------------------------------
// OnSaveTransformationPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnSaveTransformationPushButtonPressed( void )
{
  if ( ! ( m_RegAladin || m_RegNonRigid ) )
  {
    QMessageBox msgBox;
    msgBox.setText("No registration data available,"
		   "\nplease execute a registration.");
    msgBox.exec();
    
    return;
  }

  QFileDialog dialog( (QWidget *) this->parent(),
		      tr( "Save transformation") );

  dialog.setFileMode( QFileDialog::AnyFile );
  dialog.setViewMode( QFileDialog::Detail );
  dialog.setDirectory( QDir::currentPath() );
  dialog.setAcceptMode( QFileDialog::AcceptSave );
  dialog.setLabelText( QFileDialog::FileName, tr( "File selected" ) );

  QStringList filters;

  if ( ! ( m_RegAladin && m_RegNonRigid ) )
  {
    if ( m_RegNonRigid )    
    {
      filters << "B-spline control grid (*.nii)";      
      dialog.selectNameFilter( filters[1] );
    }
    else
    {
      filters << "Affine transformation (*.affine)";
      dialog.selectNameFilter( filters[0] );
    }
  }
  else {
    filters << "Affine transformation (*.affine)"
	    << "B-spline control grid (*.nii)";
    dialog.selectNameFilter( filters[0] );
  }

  dialog.setNameFilters(filters);

  if ( dialog.exec() )
  {

    // Find the specific filter used

    int i, iSelectedFilter = -1;
    QString selectedFilter = dialog.selectedNameFilter();

    for ( i = 0; i < filters.size(); i++ )
    {
      if ( filters.at(i) == selectedFilter ) 
      {
	iSelectedFilter = i;
	break;
      }
    }

    std::cout << dialog.selectedNameFilter().toStdString().c_str() << ", "
	      << iSelectedFilter 
	      << std::endl;

    // Save according to the filter selected

    switch ( iSelectedFilter )
    {
      // Affine transformation file
    case 0:
      {

	if ( ! m_RegAladin )
	{
	  QMessageBox msgBox;
	  msgBox.setText("No registration data available,"
			 "\nplease execute an affine registration.");
	  msgBox.exec();
	  
	  return;
	}

	m_RegParameters.m_AladinParameters.outputAffineName = dialog.selectedFiles()[0];

	std::cout << m_RegParameters.m_AladinParameters.outputAffineName.toStdString().c_str() 
		  << std::endl;

	reg_tool_WriteAffineFile( m_RegAladin->GetTransformationMatrix(), 
				  m_RegParameters.m_AladinParameters.outputAffineName.toStdString().c_str() );
	break;
      }

      // B-spline control grid 
    case 1:
      {

	if ( ! m_RegNonRigid )
	{
	  QMessageBox msgBox;
	  msgBox.setText("No registration data available,"
			 "\nplease execute a non-rigid registration.");
	  msgBox.exec();
	  
	  return;
	}

	m_RegParameters.m_F3dParameters.outputControlPointGridName = dialog.selectedFiles()[0];

	std::cout << m_RegParameters.m_F3dParameters.outputControlPointGridName.toStdString().c_str() 
		  << std::endl;

        nifti_image *outputControlPointGridImage 
	  = m_RegNonRigid->GetControlPointPositionImage();

        memset( outputControlPointGridImage->descrip, 0, 80 );
        strcpy ( outputControlPointGridImage->descrip,
		"Control point position from NiftyReg (reg_f3d)" );

        reg_io_WriteImageFile( outputControlPointGridImage,
			       m_RegParameters.m_F3dParameters.outputControlPointGridName.toStdString().c_str() );

        nifti_image_free( outputControlPointGridImage );

	break;
      }

      // Unrecognised filter selected
    default:
      {
	QMessageBox msgBox;
	msgBox.setText("Algorithm fault: Unrecognised SaveAs filter.");
	msgBox.exec();
      }
    }
  }
}


// ---------------------------------------------------------------------------
// WriteRegistrationParametersToFile();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::WriteRegistrationParametersToFile( QString &filename  )
{
  
  std::ofstream fout( filename.toStdString().c_str() );

  if ((! fout) || fout.bad()) {
    QMessageBox msgBox;
    msgBox.setText( QString( "ERROR: Could not open file: " ) + filename );
    msgBox.exec();
    return;
  }


  if ( m_RegParameters.m_FlagDoInitialRigidReg ) 
  {

    fout << "reg_aladin \\" << endl;

    // The reference image filename

    fout << "   -ref " 
	 << m_RegParameters.m_AladinParameters.referenceImagePath.toStdString().c_str();

    if ( ! m_RegParameters.m_AladinParameters.referenceImagePath.isEmpty() ) 
      fout << "/";

    fout << m_RegParameters.m_AladinParameters.referenceImageName.toStdString().c_str() 
	 << ".nii \\" << endl;

    // The floating image filename

    fout << "   -flo " 
	 << m_RegParameters.m_AladinParameters.floatingImagePath.toStdString().c_str();

    if ( ! m_RegParameters.m_AladinParameters.floatingImagePath.isEmpty() ) 
      fout << "/";

    fout << m_RegParameters.m_AladinParameters.floatingImageName.toStdString().c_str() 
	 << ".nii \\" << endl;

    if ( m_RegParameters.m_AladinParameters.symFlag )
      fout << "   -sym \\" << endl;

    if ( m_RegParameters.m_AladinParameters.outputAffineFlag )
      fout << "   -aff " 
	   << m_RegParameters.m_AladinParameters.outputAffineName.toStdString().c_str() 
	   << " \\" << endl;

    if ( m_RegParameters.m_AladinParameters.alignCenterFlag )
      fout << "   -nac \\" << endl;

    if ( m_RegParameters.m_AladinParameters.regnType == RIGID_ONLY )
      fout << "   -rigOnly \\" << endl;
    else if ( m_RegParameters.m_AladinParameters.regnType == DIRECT_AFFINE )
      fout << "   -affDirect \\" << endl;

    fout << "   -maxit " << m_RegParameters.m_AladinParameters.maxiterationNumber << " \\" << endl
	 << "   -%v " << m_RegParameters.m_AladinParameters.block_percent_to_use << " \\" << endl
	 << "   -%i " << m_RegParameters.m_AladinParameters.inlier_lts << " \\" << endl
	 << "   -interp " << m_RegParameters.m_AladinParameters.interpolation << " \\" << endl;

    if ( m_RegParameters.m_FlagFlirtAffine )
      fout << "   -affFlirt " << m_RegParameters.m_InputAffineName.toStdString().c_str() << " \\" << endl;
    else if ( m_RegParameters.m_FlagInputAffine )
      fout << "   -inaff " << m_RegParameters.m_InputAffineName.toStdString().c_str() << " \\" << endl;

    // The reference image mask filename

    if ( ! m_RegParameters.m_AladinParameters.referenceMaskName.isEmpty() ) {
      fout << "   -rmask " 
	   << m_RegParameters.m_AladinParameters.referenceMaskName.toStdString().c_str();

      if ( ! m_RegParameters.m_AladinParameters.referenceMaskName.isEmpty() ) 
	fout << "/";
      
      fout << m_RegParameters.m_AladinParameters.referenceMaskName.toStdString().c_str() 
	   << ".nii \\" << endl;
    }

    if ( m_RegParameters.m_TargetSigmaValue )
      fout << "   -smooR " << m_RegParameters.m_TargetSigmaValue << " \\" << endl;

    if ( m_RegParameters.m_SourceSigmaValue )
      fout << "   -smooF " << m_RegParameters.m_SourceSigmaValue << " \\" << endl;

    fout << "   -ln " << m_RegParameters.m_LevelNumber << " \\" << endl
	 << "   -lp " << m_RegParameters.m_Level2Perform << " \\" << endl;

    if ( m_RegParameters.m_AladinParameters.outputResultFlag ) {
      fout << "   -res " 
	   << m_RegParameters.m_AladinParameters.outputResultPath.toStdString().c_str();

      if ( ! m_RegParameters.m_AladinParameters.outputResultPath.isEmpty() ) 
	fout << "/";
      
      fout  << m_RegParameters.m_AladinParameters.outputResultName.toStdString().c_str() 
	    << ".nii" << endl << endl;
    }
    else
      fout << "   -res outputAffineResult.nii" << endl << endl;
  }


  if ( m_RegParameters.m_FlagDoNonRigidReg ) 
  {
    fout << "reg_f3d \\" << endl;

    // The reference image filename

    fout << "   -ref " 
	 << m_RegParameters.m_F3dParameters.referenceImagePath.toStdString().c_str();
    
    if ( ! m_RegParameters.m_F3dParameters.referenceImagePath.isEmpty() ) 
      fout << "/";
      
    fout << m_RegParameters.m_F3dParameters.referenceImageName.toStdString().c_str()
	 << ".nii \\" << endl;

    // The floating image filename

    fout << "   -flo "
	 << m_RegParameters.m_F3dParameters.floatingImagePath.toStdString().c_str();
    
    if ( ! m_RegParameters.m_F3dParameters.floatingImagePath.isEmpty() ) 
      fout << "/";
      
    fout << m_RegParameters.m_F3dParameters.floatingImageName.toStdString().c_str()
	 << ".nii \\" << endl;

    if ( m_RegParameters.m_F3dParameters.inputControlPointGridFlag )
      fout << "   -incpp " << m_RegParameters.m_F3dParameters.inputControlPointGridName.toStdString().c_str() << ".nii \\" << endl;
    
    if ( m_RegParameters.m_FlagFlirtAffine )
      fout << "   -affFlirt " << m_RegParameters.m_InputAffineName.toStdString().c_str() << " \\" << endl;
    else if ( m_RegParameters.m_FlagInputAffine )
      fout << "   -inaff " << m_RegParameters.m_InputAffineName.toStdString().c_str() << " \\" << endl;
    
    if ( ! m_RegParameters.m_F3dParameters.outputControlPointGridName.isEmpty() )
      fout << "   -cpp " << m_RegParameters.m_F3dParameters.outputControlPointGridName.toStdString().c_str() << " \\" << endl;
    
    // The reference image mask name
    
    if ( ! m_RegParameters.m_F3dParameters.referenceMaskName.isEmpty() ) {
      fout << "   -rmask " 
	   << m_RegParameters.m_F3dParameters.referenceMaskPath.toStdString().c_str();
      
      if ( ! m_RegParameters.m_F3dParameters.referenceMaskPath.isEmpty() ) 
	fout << "/";
      
      fout << m_RegParameters.m_F3dParameters.referenceMaskName.toStdString().c_str() 
	   << ".nii \\" << endl;
    }

    if ( m_RegParameters.m_TargetSigmaValue )
      fout << "   -smooR " << m_RegParameters.m_TargetSigmaValue << " \\" << endl;
    
    if ( m_RegParameters.m_SourceSigmaValue )
      fout << "   -smooF " << m_RegParameters.m_SourceSigmaValue << " \\" << endl;
    
    fout << "   -rbn " << m_RegParameters.m_F3dParameters.referenceBinNumber << " \\" << endl
	 << "   -fbn " << m_RegParameters.m_F3dParameters.floatingBinNumber << " \\" << endl;
    
    if ( m_RegParameters.m_F3dParameters.referenceThresholdUp != -std::numeric_limits<PrecisionTYPE>::max() )
      fout << "   -rUpTh " << m_RegParameters.m_F3dParameters.referenceThresholdUp << " \\" << endl;
    if ( m_RegParameters.m_F3dParameters.referenceThresholdLow != -std::numeric_limits<PrecisionTYPE>::max() )
      fout << "   -rLwTh " << m_RegParameters.m_F3dParameters.referenceThresholdLow << " \\" << endl;
    
    if ( m_RegParameters.m_F3dParameters.floatingThresholdUp != -std::numeric_limits<PrecisionTYPE>::max() )
      fout << "   -fUpTh " << m_RegParameters.m_F3dParameters.floatingThresholdUp << " \\" << endl;
    if ( m_RegParameters.m_F3dParameters.floatingThresholdLow != -std::numeric_limits<PrecisionTYPE>::max() )
      fout << "   -fLwTh " << m_RegParameters.m_F3dParameters.floatingThresholdLow << " \\" << endl;
    
    fout << "   -sx " << m_RegParameters.m_F3dParameters.spacing[0] << " \\" << endl
	 << "   -sy " << m_RegParameters.m_F3dParameters.spacing[1] << " \\" << endl
	 << "   -sz " << m_RegParameters.m_F3dParameters.spacing[2] << " \\" << endl;
    
    fout << "   -be " << m_RegParameters.m_F3dParameters.bendingEnergyWeight << " \\" << endl
      
	 << "   -le " << m_RegParameters.m_F3dParameters.linearEnergyWeight0 
	 << " " << m_RegParameters.m_F3dParameters.linearEnergyWeight1 << " \\" << endl
      
	 << "   -jl " << m_RegParameters.m_F3dParameters.jacobianLogWeight << " \\" << endl;
    
    if ( ! m_RegParameters.m_F3dParameters.jacobianLogApproximation )
      fout << "   -noAppJL \\" << endl;
    
    if ( ! m_RegParameters.m_F3dParameters.useConjugate )
      fout << "   -noConj \\" << endl;
    
    if ( m_RegParameters.m_F3dParameters.similarity == SSD_SIMILARITY )
      fout << "   -ssd \\" << endl;
    else if ( m_RegParameters.m_F3dParameters.similarity == KLDIV_SIMILARITY )
      fout << "   -kld \\" << endl;
    
    fout << "   -maxit " << m_RegParameters.m_F3dParameters.maxiterationNumber << " \\" << endl;
    
    fout << "   -ln " << m_RegParameters.m_LevelNumber << " \\" << endl
	 << "   -lp " << m_RegParameters.m_Level2Perform << " \\" << endl;
    
    if ( m_RegParameters.m_F3dParameters.noPyramid )
      fout << "   -nopy \\" << endl;
   
    fout << "   -interp " << m_RegParameters.m_F3dParameters.interpolation << " \\" << endl;
 
    if ( m_RegParameters.m_F3dParameters.gradientSmoothingSigma )
      fout << "   -smoothGrad " << m_RegParameters.m_F3dParameters.gradientSmoothingSigma << " \\" << endl;
    
    if ( m_RegParameters.m_F3dParameters.warpedPaddingValue != -std::numeric_limits<PrecisionTYPE>::max() )
      fout << "   -pad " << m_RegParameters.m_F3dParameters.warpedPaddingValue << " \\" << endl;
    
    if ( ! m_RegParameters.m_F3dParameters.verbose )
      fout << "   -voff \\" << endl;
    
    if ( ! m_RegParameters.m_F3dParameters.outputWarpedName.isEmpty() )
    {
      fout << "   -res " 
	   << m_RegParameters.m_F3dParameters.outputWarpedPath.toStdString().c_str();
      
      if ( ! m_RegParameters.m_F3dParameters.outputWarpedPath.isEmpty() ) 
	fout << "/";
      
      fout << m_RegParameters.m_F3dParameters.outputWarpedName.toStdString().c_str() 
	   << ".nii" << endl << endl;
    }
    else
      fout << "   -res outputNonRigidResult.nii" << endl << endl;    
  }

 
  fout.close();
}


// ---------------------------------------------------------------------------
// OnSaveRegistrationParametersPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnSaveRegistrationParametersPushButtonPressed( void )
{

  QFileDialog dialog( (QWidget *) this->parent(),
		      tr( "Save registration parameters") );

  dialog.setFileMode( QFileDialog::AnyFile );
  dialog.setViewMode( QFileDialog::Detail );
  dialog.setDirectory( QDir::currentPath() );
  dialog.setAcceptMode( QFileDialog::AcceptSave );
  dialog.setLabelText( QFileDialog::FileName, tr( "File selected" ) );

  QStringList filters;
  filters << "Registration parameters (*.sh)";

  dialog.setNameFilters(filters);

  if ( dialog.exec() )
    WriteRegistrationParametersToFile( dialog.selectedFiles()[0] );
}


// ---------------------------------------------------------------------------
// ReadRegistrationParametersFromFile();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::ReadRegistrationParametersFromFile( QString &filename )
{
  typedef enum { UNSET, REG_ALADIN, REG_F3D } inParamsEnumType;
  inParamsEnumType inParametersType = UNSET;

  std::string inString;
  
  NiftyRegParameters<PrecisionTYPE> inRegParameters;

  // Although 'symFlag=true' is the default for the GUI, option '-sym'
  // sets the flag to be true so we initialise it here to be false.
  inRegParameters.m_AladinParameters.symFlag = false;


  std::ifstream fin( filename.toStdString().c_str() );

  if ((! fin) || fin.bad()) 
  {
    QMessageBox msgBox;
    msgBox.setText( QString( "ERROR: Could not open file: " ) + filename );
    msgBox.exec();
    return;
  }

  while ( ( ! fin.eof() ) && ( inParametersType == UNSET ) ) 
  {
    
    fin >> inString;
    
    if ( inString.compare( std::string( "reg_aladin" ) ) == 0 ) 
    {
      inParametersType = REG_ALADIN;
    }
    else if ( inString.compare( std::string( "reg_f3d" ) ) == 0 ) 
    {
      inParametersType = REG_F3D;
    }
  }

  
  if ( inParametersType == UNSET )
  {
    QMessageBox msgBox;
    msgBox.setText( QString( "ERROR: Could not read file: " ) + filename );
    msgBox.exec();
    return;    
  }

  while ( ! fin.eof() ) 
  {
    
    fin >> inString;
    
    // Read the Aladin parameters

    if ( inParametersType == REG_ALADIN )
    {

      if ( inString.compare( std::string( "reg_aladin" ) ) == 0 ) 
      {
	QMessageBox msgBox;
	msgBox.setText( "ERROR: Duplicate 'reg_aladin' string found." );
	msgBox.exec();
	return;    
      }

      else if ( inString.compare( std::string( "reg_f3d" ) ) == 0 ) 
      {
	inParametersType = REG_F3D;
      }

      else if ( inString.compare( std::string( "-ref" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_AladinParameters.referenceImagePath 
	  = QString( itksys::SystemTools::GetParentDirectory( inString.c_str() ).c_str() );
	inRegParameters.m_AladinParameters.referenceImageName 
	  = QString( itksys::SystemTools::GetFilenameName( inString ).c_str() );
      }
      else if ( inString.compare( std::string( "-flo" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_AladinParameters.floatingImagePath 
	  = QString( itksys::SystemTools::GetParentDirectory( inString.c_str() ).c_str() );
	inRegParameters.m_AladinParameters.floatingImageName 
	  = QString( itksys::SystemTools::GetFilenameName( inString ).c_str() );
      }
      
      else if ( inString.compare( std::string( "-sym" ) ) == 0 ) 
      {
	inRegParameters.m_AladinParameters.symFlag = true;
      }

      else if ( inString.compare( std::string( "-aff" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_AladinParameters.outputAffineName = QString( inString.c_str() );
	inRegParameters.m_AladinParameters.outputAffineFlag  = true;
      }
      
      else if ( inString.compare( std::string( "-nac" ) ) == 0 ) 
      {
	inRegParameters.m_AladinParameters.alignCenterFlag = false;
      }

      else if ( inString.compare( std::string( "-rigOnly" ) ) == 0 ) 
      {
	inRegParameters.m_AladinParameters.regnType = RIGID_ONLY;
      }
      else if ( inString.compare( std::string( "-affDirect" ) ) == 0 ) 
      {
	inRegParameters.m_AladinParameters.regnType = DIRECT_AFFINE;
      }
      else if ( inString.compare( std::string( "-maxit" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_AladinParameters.maxiterationNumber = atoi( inString.c_str() );
      }
      else if ( inString.compare( std::string( "-%v" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_AladinParameters.block_percent_to_use = atoi( inString.c_str() );
      }
      else if ( inString.compare( std::string( "-%i" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_AladinParameters.inlier_lts = atoi( inString.c_str() );
      }

      else if ( inString.compare( std::string( "-interp" ) ) == 0 ) 
      {
	fin >> inString;
	switch( atoi( inString.c_str() ) )
	{
	case 1: 
	{
	  inRegParameters.m_AladinParameters.interpolation = NEAREST_INTERPOLATION;
	  break;
	}
	case 2: 
	{
	  inRegParameters.m_AladinParameters.interpolation = LINEAR_INTERPOLATION;
	  break;
	}
	case 3: 
	{
	  inRegParameters.m_AladinParameters.interpolation = CUBIC_INTERPOLATION;
	  break;
	}
	default: 
	{
	  QMessageBox msgBox;
	  msgBox.setText( QString( "ERROR: Interpolation method unrecognised" ) );
	  msgBox.exec();
	  return;
	}
	}       
      }

      else if ( inString.compare( std::string( "-affFlirt" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_InputAffineName = QString( inString.c_str() );
	inRegParameters.m_FlagInputAffine = true;
	inRegParameters.m_FlagFlirtAffine = true;
      }
      else if ( inString.compare( std::string( "-inaff" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_InputAffineName = QString( inString.c_str() );
	inRegParameters.m_FlagInputAffine = true;
      }

      else if ( inString.compare( std::string( "-rmask" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_AladinParameters.referenceMaskPath
	  = QString( itksys::SystemTools::GetParentDirectory( inString.c_str() ).c_str() );
	inRegParameters.m_AladinParameters.referenceMaskName 
	  = QString( itksys::SystemTools::GetFilenameName( inString ).c_str() );
      }

      else if ( inString.compare( std::string( "-smooR" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_TargetSigmaValue = atof( inString.c_str() ); 
      }

      else if ( inString.compare( std::string( "-smooF" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_SourceSigmaValue = atof( inString.c_str() ); 
      }

      else if ( inString.compare( std::string( "-ln" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_LevelNumber = atoi( inString.c_str() ); 
      }

      else if ( inString.compare( std::string( "-lp" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_Level2Perform = atoi( inString.c_str() ); 
      }

      else if ( inString.compare( std::string( "-res" ) ) == 0 ) 
      {
	inRegParameters.m_AladinParameters.outputResultFlag = true;
	fin >> inString;
	inRegParameters.m_AladinParameters.outputResultPath
	  = QString( itksys::SystemTools::GetParentDirectory( inString.c_str() ).c_str() );
	inRegParameters.m_AladinParameters.outputResultName 
	  = QString( itksys::SystemTools::GetFilenameName( inString ).c_str() );
      }

      else if ( inString.compare( std::string( "\\" ) ) != 0 ) 
      {
	QMessageBox msgBox;
	msgBox.setText( QString( "ERROR: Unrecognised field " ) + inString.c_str() );
	msgBox.exec();
	return;
      }
    }


    // Read the Non-Rigid parameters

    else if ( inParametersType == REG_F3D )
    {


      if ( inString.compare( std::string( "-ref" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.referenceImagePath 
	  = QString( itksys::SystemTools::GetParentDirectory( inString.c_str() ).c_str() );
	inRegParameters.m_F3dParameters.referenceImageName 
	  = QString( itksys::SystemTools::GetFilenameName( inString ).c_str() );
      }
      else if ( inString.compare( std::string( "-flo" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.floatingImagePath 
	  = QString( itksys::SystemTools::GetParentDirectory( inString.c_str() ).c_str() );
	inRegParameters.m_F3dParameters.floatingImageName 
	  = QString( itksys::SystemTools::GetFilenameName( inString ).c_str() );
      }

      else if ( inString.compare( std::string( "-incpp" ) ) == 0 ) 
      {
	inRegParameters.m_F3dParameters.inputControlPointGridFlag = true;
	fin >> inString;
	inRegParameters.m_F3dParameters.inputControlPointGridName = QString( inString.c_str() );
      }
          

      else if ( inString.compare( std::string( "-affFlirt" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_InputAffineName = QString( inString.c_str() );
	inRegParameters.m_FlagInputAffine = true;
	inRegParameters.m_FlagFlirtAffine = true;
      }
      else if ( inString.compare( std::string( "-inaff" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_InputAffineName = QString( inString.c_str() );
	inRegParameters.m_FlagInputAffine = true;
      }
      
      else if ( inString.compare( std::string( "-cpp" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.outputControlPointGridName = QString( inString.c_str() );
      }

      else if ( inString.compare( std::string( "-rmask" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.referenceMaskPath
	  = QString( itksys::SystemTools::GetParentDirectory( inString.c_str() ).c_str() );
	inRegParameters.m_F3dParameters.referenceMaskName 
	  = QString( itksys::SystemTools::GetFilenameName( inString ).c_str() );
      }
      else if ( inString.compare( std::string( "-smooR" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_TargetSigmaValue = atof( inString.c_str() ); 
      }

      else if ( inString.compare( std::string( "-smooF" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_SourceSigmaValue = atof( inString.c_str() ); 
      }
    
      else if ( inString.compare( std::string( "-rbn" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.referenceBinNumber = atoi( inString.c_str() ); 
      }

      else if ( inString.compare( std::string( "-fbn" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.floatingBinNumber  = atoi( inString.c_str() ); 
      }
    
      else if ( inString.compare( std::string( "-rUpTh" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.referenceThresholdUp = atof( inString.c_str() ); 
      }
      else if ( inString.compare( std::string( "-rLwTh" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.referenceThresholdLow = atof( inString.c_str() ); 
      }
      
      else if ( inString.compare( std::string( "-fUpTh" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.floatingThresholdUp = atof( inString.c_str() ); 
      }
      else if ( inString.compare( std::string( "-fLwTh" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.floatingThresholdLow = atof( inString.c_str() ); 
      }
    
      else if ( inString.compare( std::string( "-sx" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.spacing[0] = atof( inString.c_str() ); 
      }
      else if ( inString.compare( std::string( "-sy" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.spacing[1] = atof( inString.c_str() ); 
      }
      else if ( inString.compare( std::string( "-sz" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.spacing[2] = atof( inString.c_str() ); 
      }
    
      else if ( inString.compare( std::string( "-be" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.bendingEnergyWeight = atof( inString.c_str() ); 
      }
      
      else if ( inString.compare( std::string( "-le" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.linearEnergyWeight0 = atof( inString.c_str() ); 
	fin >> inString;
	inRegParameters.m_F3dParameters.linearEnergyWeight1 = atof( inString.c_str() ); 
      }

      
      else if ( inString.compare( std::string( "-jl" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.jacobianLogWeight = atof( inString.c_str() ); 
      }

      else if ( inString.compare( std::string( "-noAppJL" ) ) == 0 ) 
      {
	inRegParameters.m_F3dParameters.jacobianLogApproximation = false;
      }    
      else if ( inString.compare( std::string( "-noConj" ) ) == 0 ) 
      {
	inRegParameters.m_F3dParameters.useConjugate = false;
      }
    
      else if ( inString.compare( std::string( "-ssd" ) ) == 0 ) 
      {
	inRegParameters.m_F3dParameters.similarity = SSD_SIMILARITY;
      }    
      else if ( inString.compare( std::string( "-kld" ) ) == 0 ) 
      {
	inRegParameters.m_F3dParameters.similarity = KLDIV_SIMILARITY;
      }
    
      else if ( inString.compare( std::string( "-maxit" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.maxiterationNumber = atoi( inString.c_str() ); 
      }
    
     else if ( inString.compare( std::string( "-ln" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_LevelNumber = atoi( inString.c_str() ); 
      }

      else if ( inString.compare( std::string( "-lp" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_Level2Perform = atoi( inString.c_str() ); 
      }
    
      else if ( inString.compare( std::string( "-nopy" ) ) == 0 ) 
      {
	inRegParameters.m_F3dParameters.noPyramid = true;
      }
    
      else if ( inString.compare( std::string( "-interp" ) ) == 0 ) 
      {
	fin >> inString;
	switch( atoi( inString.c_str() ) )
	{
	case 1: 
	{
	  inRegParameters.m_F3dParameters.interpolation = NEAREST_INTERPOLATION;
	  break;
	}
	case 2: 
	{
	  inRegParameters.m_F3dParameters.interpolation = LINEAR_INTERPOLATION;
	  break;
	}
	case 3: 
	{
	  inRegParameters.m_F3dParameters.interpolation = CUBIC_INTERPOLATION;
	  break;
	}
	default: 
	{
	  QMessageBox msgBox;
	  msgBox.setText( QString( "ERROR: Interpolation method unrecognised" ) );
	  msgBox.exec();
	  return;
	}
	}       
      }

      else if ( inString.compare( std::string( "-smoothGrad" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.gradientSmoothingSigma = atof( inString.c_str() ); 
      }
    
      else if ( inString.compare( std::string( "-pad" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.warpedPaddingValue =  atof( inString.c_str() ); 
      }

      else if ( inString.compare( std::string( "-voff" ) ) == 0 ) 
      {
	inRegParameters.m_F3dParameters.verbose = false;
      }
    
      else if ( inString.compare( std::string( "-res" ) ) == 0 ) 
      {
	fin >> inString;
	inRegParameters.m_F3dParameters.outputWarpedPath 
	  = QString( itksys::SystemTools::GetParentDirectory( inString.c_str() ).c_str() );
	inRegParameters.m_F3dParameters.outputWarpedName 
	  = QString( itksys::SystemTools::GetFilenameName( inString ).c_str() );
      }
    }
  }

  m_RegParameters = inRegParameters;
  SetGuiToParameterValues();

  fin.close();
}


// ---------------------------------------------------------------------------
// OnLoadRegistrationParametersPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnLoadRegistrationParametersPushButtonPressed( void )
{

  QFileDialog dialog( (QWidget *) this->parent(),
		      tr( "Load registration parameters") );

  dialog.setFileMode( QFileDialog::AnyFile );
  dialog.setViewMode( QFileDialog::Detail );
  dialog.setDirectory( QDir::currentPath() );
  dialog.setAcceptMode( QFileDialog::AcceptOpen );
  dialog.setLabelText( QFileDialog::FileName, tr( "File selected" ) );

  QStringList filters;
  filters << "Registration parameters (*.sh)";

  dialog.setNameFilters(filters);

  if ( dialog.exec() )
    ReadRegistrationParametersFromFile( dialog.selectedFiles()[0] );
}


// ---------------------------------------------------------------------------
// OnExecutePushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnExecutePushButtonPressed( void )
{

#if 0

  // This is to test the image conversion from MITK to nifti and back

  mitk::DataStorage::SetOfObjects::ConstPointer nodes = GetNodes();

  if ( ( nodes ) && ( nodes->size() >= 1 ) )
  {

    mitk::Image::Pointer mitkSourceImage = dynamic_cast<mitk::Image*>( (*nodes)[0]->GetData() );

    cout << "Creating nifti image" << endl;
    nifti_image *niftiSourceImage  = ConvertMitkImageToNifti( mitkSourceImage );

    nifti_set_filenames( niftiSourceImage,
			 "/scratch0/NOT_BACKED_UP/JamiesAffineRegnHeaderTestData/testNiftiFromItk.nii",0,0 );
    nifti_image_write( niftiSourceImage );

    cout << "Creating mitk image" << endl;
    mitkSourceImage = ConvertNiftiImageToMitk( niftiSourceImage );
    
    nifti_image_free( niftiSourceImage );

    cout << "Adding mitk image to data storage" << endl;
    mitk::DataNode::Pointer resultNode = mitk::DataNode::New();
    resultNode->SetProperty("name", mitk::StringProperty::New( "test" ) );
    resultNode->SetData( mitkSourceImage );

    GetDataStorage()->Add( resultNode );

    cout << "Done" << endl;
  }

  return;

#endif



  PrintSelf( std::cout );

  if ( ! m_Modified )
  {
    //return;
  }


#ifdef USE_THREADING

  itk::MultiThreader::Pointer threader = itk::MultiThreader::New();

  itk::ThreadFunctionType pointer = &ExecuteRegistration;
  threader->SpawnThread( pointer, this );

#else
#ifdef USE_QT_THREADING

  QEventLoop q;
  RegistrationExecution regExecutionThread( this );

#if 0
  // The cancel button terminates the registration
  connect( m_Controls.m_CancelPushButton, SIGNAL( pressed( ) ), 
	   &regExecutionThread, SLOT( quit( ) ) );
#endif

  // The event loop 'q' terminates when the registration finishes (or is terminated)
  connect( &regExecutionThread, SIGNAL( finished( ) ), 
	   &q, SLOT( quit( ) ) );

  regExecutionThread.start();
  q.exec();

#else

  ExecuteRegistration( this );

#endif
#endif

}


// ---------------------------------------------------------------------------
// CreateDeformationVisualisationSurface();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::CreateDeformationVisualisationSurface( void )
{
  if ( ! m_RegNonRigid )
  {
    QMessageBox msgBox;
    msgBox.setText("No registration data available,"
		   "\nplease execute a non-rigid registration.");
    msgBox.exec();
    
    return;
  }

  nifti_image *controlPointGrid 
    = m_RegNonRigid->GetControlPointPositionImage();

  std::cout << "Control point grid dimensions: " 
	    << controlPointGrid->nx << " x " 
	    << controlPointGrid->ny << " x " 
	    << controlPointGrid->nz << std::endl
	    << "Control point grid spacing: " 
	    << controlPointGrid->dx << " x " 
	    << controlPointGrid->dy << " x " 
	    << controlPointGrid->dz << std::endl;



  if ( this->controlPointGrid != NULL )
    nifti_image_free( controlPointGrid );
}


// ---------------------------------------------------------------------------
// ExecuteRegistration();
// --------------------------------------------------------------------------- 

ITK_THREAD_RETURN_TYPE ExecuteRegistration( void *param )
{
#ifdef _USE_CUDA
  std::cout << "USING CUDA" << std::endl;
#else
  std::cout << "NOT USING CUDA" << std::endl;
#endif

#ifdef USE_THREADING

  std::cout<<"Getting thread info data... ";
  itk::MultiThreader::ThreadInfoStruct* threadInfo;
  threadInfo = static_cast<itk::MultiThreader::ThreadInfoStruct*>(param);

  if (!threadInfo) 
  {
    std::cout<<"[FAILED]"<<std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout<<"[PASSED]"<<std::endl;

  
  std::cout<<"Getting user data from thread... ";
  QmitkNiftyRegView* userData = static_cast<QmitkNiftyRegView*>(threadInfo->UserData);

  if (!userData) 
  {
    std::cout<<"[FAILED]"<<std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout<<"[PASSED]"<<std::endl;

#else

  QmitkNiftyRegView* userData = static_cast<QmitkNiftyRegView*>( param );

#endif

  // Get the source and target MITK images from the data manager

  std::string name;

  QString sourceName = userData->m_Controls.m_SourceImageComboBox->currentText();
  QString targetName = userData->m_Controls.m_TargetImageComboBox->currentText();

  QString targetMaskName = userData->m_Controls.m_TargetMaskImageComboBox->currentText();

  mitk::Image::Pointer mitkSourceImage = 0;
  mitk::Image::Pointer mitkTransformedImage = 0;
  mitk::Image::Pointer mitkTargetImage = 0;

  mitk::Image::Pointer mitkTargetMaskImage = 0;


  mitk::DataStorage::SetOfObjects::ConstPointer nodes = userData->GetNodes();

  mitk::DataNode::Pointer nodeSource = 0;
  mitk::DataNode::Pointer nodeTarget = 0;
  
  if ( nodes ) 
  {
    if (nodes->size() > 0) 
    {
      for (unsigned int i=0; i<nodes->size(); i++) 
      {
	(*nodes)[i]->GetStringProperty("name", name);

	if ( ( QString(name.c_str()) == sourceName ) && ( ! mitkSourceImage ) ) 
	{
	  mitkSourceImage = dynamic_cast<mitk::Image*>((*nodes)[i]->GetData());
	  nodeSource = (*nodes)[i];
	}

	else if ( ( QString(name.c_str()) == targetName ) && ( ! mitkTargetImage ) ) 
	{
	  mitkTargetImage = dynamic_cast<mitk::Image*>((*nodes)[i]->GetData());
	  nodeTarget = (*nodes)[i];
	}

	else if ( ( QString(name.c_str()) == targetMaskName ) && ( ! mitkTargetMaskImage ) )
	  mitkTargetMaskImage = dynamic_cast<mitk::Image*>((*nodes)[i]->GetData());
      }
    }
  }

  
  // Ensure the progress bar is scaled appropriately

  if ( userData->m_RegParameters.m_FlagDoInitialRigidReg && userData->m_RegParameters.m_FlagDoNonRigidReg ) 
    userData->m_ProgressBarRange = 50.;
  else
    userData->m_ProgressBarRange = 100.;

  userData->m_ProgressBarOffset = 0.;

  
  // Delete the previous registrations
  
  userData->m_RegParameters.DeallocateImages();

  if ( userData->m_RegAladin ) 
  {
    delete userData->m_RegAladin;
    userData->m_RegAladin = 0;
  }

  if ( userData->m_RegNonRigid )
  {
    delete userData->m_RegNonRigid;
    userData->m_RegNonRigid = 0;
  }


  // Create and run the Aladin registration?

  if ( userData->m_RegParameters.m_FlagDoInitialRigidReg ) 
  {

    userData->m_RegAladin = 
      userData->m_RegParameters.CreateAladinRegistrationObject( mitkSourceImage, 
								mitkTargetImage, 
								mitkTargetMaskImage );
  
    userData->m_RegAladin->SetProgressCallbackFunction( &UpdateProgressBar, userData );

    userData->m_RegAladin->Run();

    mitkSourceImage = ConvertNiftiImageToMitk( userData->m_RegAladin->GetFinalWarpedImage() );

    // Add this result to the data manager
    mitk::DataNode::Pointer resultNode = mitk::DataNode::New();

    std::string nameOfResultImage;
    if ( userData->m_RegParameters.m_AladinParameters.regnType == RIGID_ONLY )
      nameOfResultImage = "rigid registration to ";
    else
      nameOfResultImage = "affine registration to ";
    nameOfResultImage.append( nodeTarget->GetName() );

    resultNode->SetProperty("name", mitk::StringProperty::New(nameOfResultImage) );
    resultNode->SetData( mitkSourceImage );

    userData->GetDataStorage()->Add( resultNode, nodeSource );

    UpdateProgressBar( 100., userData );

    if ( userData->m_RegParameters.m_FlagDoNonRigidReg ) 
      userData->m_ProgressBarOffset = 50.;
  }


  // Create and run the F3D registration

  if ( userData->m_RegParameters.m_FlagDoNonRigidReg ) 
  {

    userData->m_RegNonRigid = 
      userData->m_RegParameters.CreateNonRigidRegistrationObject( mitkSourceImage, 
								  mitkTargetImage, 
								  mitkTargetMaskImage );  

    userData->m_RegNonRigid->SetProgressCallbackFunction( &UpdateProgressBar, 
							  userData );

    userData->m_RegNonRigid->Run_f3d();

    mitkTransformedImage = ConvertNiftiImageToMitk( userData->m_RegNonRigid->GetWarpedImage()[0] );

    // Add this result to the data manager
    mitk::DataNode::Pointer resultNode = mitk::DataNode::New();

    std::string nameOfResultImage( "non-rigid registration to " );
    nameOfResultImage.append( nodeTarget->GetName() );

    resultNode->SetProperty("name", mitk::StringProperty::New(nameOfResultImage) );
    resultNode->SetData( mitkTransformedImage );

    userData->GetDataStorage()->Add( resultNode, nodeSource );

    UpdateProgressBar( 100., userData );

   }


  userData->m_Modified = false;
  userData->m_Controls.m_ExecutePushButton->setEnabled( false );

  return ITK_THREAD_RETURN_VALUE;
}


// ---------------------------------------------------------------------------
// UpdateProgressBar();
// --------------------------------------------------------------------------- 

void UpdateProgressBar( float pcntProgress, void *param )
{
  QmitkNiftyRegView* userData = static_cast<QmitkNiftyRegView*>( param );
  
  userData->m_Controls.m_ProgressBar->setValue( (int) (userData->m_ProgressBarOffset 
						       + userData->m_ProgressBarRange*pcntProgress/100.) );

  QCoreApplication::processEvents();

}


// ---------------------------------------------------------------------------
// OnNumberOfLevelsSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNumberOfLevelsSpinBoxValueChanged(int value)
{
  m_RegParameters.m_LevelNumber = value;

  if ( m_RegParameters.m_LevelNumber < m_RegParameters.m_Level2Perform )
  {
    m_RegParameters.m_Level2Perform = m_RegParameters.m_LevelNumber;
    m_Controls.m_LevelsToPerformSpinBox->setValue( m_RegParameters.m_Level2Perform );
  }

  Modified();
}


// ---------------------------------------------------------------------------
// OnLevelsToPerformSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnLevelsToPerformSpinBoxValueChanged(int value)
{
  m_RegParameters.m_Level2Perform = value;

  if ( m_RegParameters.m_Level2Perform > m_RegParameters.m_LevelNumber )
  {
    m_RegParameters.m_LevelNumber = m_RegParameters.m_Level2Perform;
    m_Controls.m_NumberOfLevelsSpinBox->setValue( m_RegParameters.m_LevelNumber );
  }

  Modified();
}


// ---------------------------------------------------------------------------
// OnSmoothSourceImageDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnSmoothSourceImageDoubleSpinBoxValueChanged(double value)
{
  m_RegParameters.m_SourceSigmaValue = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnSmoothTargetImageDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnSmoothTargetImageDoubleSpinBoxValueChanged(double value)
{
  m_RegParameters.m_TargetSigmaValue = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnNoSmoothingPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNoSmoothingPushButtonPressed( void )
{
  m_RegParameters.m_SourceSigmaValue = 0.;
  m_Controls.m_SmoothSourceImageDoubleSpinBox->setValue( m_RegParameters.m_SourceSigmaValue );

  m_RegParameters.m_TargetSigmaValue = 0.;
  m_Controls.m_SmoothTargetImageDoubleSpinBox->setValue( m_RegParameters.m_TargetSigmaValue );

  Modified();
}


// ---------------------------------------------------------------------------
// OnDoBlockMatchingOnlyRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnDoBlockMatchingOnlyRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegParameters.m_FlagDoInitialRigidReg = true;
    m_RegParameters.m_FlagDoNonRigidReg = false;
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnDoNonRigidOnlyRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnDoNonRigidOnlyRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegParameters.m_FlagDoInitialRigidReg = false;
    m_RegParameters.m_FlagDoNonRigidReg = true;
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnDoBlockMatchingThenNonRigidRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnDoBlockMatchingThenNonRigidRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegParameters.m_FlagDoInitialRigidReg = true;
    m_RegParameters.m_FlagDoNonRigidReg = true;
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnInputAffineCheckBoxStateChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnInputAffineCheckBoxToggled( bool checked )
{
  m_RegParameters.m_FlagInputAffine = checked;
  Modified();
}


// ---------------------------------------------------------------------------
// OnInputAffineBrowsePushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnInputAffineBrowsePushButtonPressed( void )
{
  m_RegParameters.m_InputAffineName = 
    QFileDialog::getOpenFileName(NULL, 
				 tr("Select affine transformation file"), 
				 QDir::currentPath(), 
				 tr("Affine transform file (*.txt *.tfm);;"
				    "Any file (*)"));

  if ( ! m_RegParameters.m_InputAffineName.isEmpty() )
  {
    m_RegParameters.m_FlagInputAffine = true;
    m_Controls.m_InputAffineFileNameLineEdit->setText( m_RegParameters.m_InputAffineName );

  }
  else
  {
    m_RegParameters.m_FlagInputAffine = false;
    m_Controls.m_InputAffineFileNameLineEdit->setText( QString( "" ) );
  }

  m_Controls.m_InitialAffineTransformationGroupBox->setChecked( m_RegParameters.m_FlagInputAffine );

  Modified();
}


// ---------------------------------------------------------------------------
// OnInputFlirtCheckBoxStateChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnInputFlirtCheckBoxStateChanged( int state )
{
  if ( state == Qt::Checked )
  {
    m_RegParameters.m_FlagFlirtAffine = true;
  }
  else
  {
    m_RegParameters.m_FlagFlirtAffine = false;
  }    
  Modified();
}


// ---------------------------------------------------------------------------
// OnUseNiftyHeaderCheckBoxStateChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnUseNiftyHeaderCheckBoxStateChanged( int state )
{
  if ( state == Qt::Checked )
  {
    // Use the nifti header origins to initialise the translation
    m_RegParameters.m_AladinParameters.alignCenterFlag = false;
  }
  else
  {
    m_RegParameters.m_AladinParameters.alignCenterFlag = true;
  }    
  Modified();
}


// ---------------------------------------------------------------------------
// OnRigidOnlyRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnRigidOnlyRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegParameters.m_AladinParameters.regnType = RIGID_ONLY;
    UpdateAladinResultImageFilename();
    UpdateNonRigidResultImageFilename();
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnRigidThenAffineRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnRigidThenAffineRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegParameters.m_AladinParameters.regnType = RIGID_THEN_AFFINE;
    UpdateAladinResultImageFilename();
    UpdateNonRigidResultImageFilename();
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnDirectAffineRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnDirectAffineRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegParameters.m_AladinParameters.regnType = DIRECT_AFFINE;
    UpdateAladinResultImageFilename();
    UpdateNonRigidResultImageFilename();
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnAladinUseSymmetricAlgorithmCheckBoxStateChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnAladinUseSymmetricAlgorithmCheckBoxStateChanged( int state )
{
  if ( state == Qt::Checked )
  {
    m_RegParameters.m_AladinParameters.symFlag = true;
  }
  else
  {
    m_RegParameters.m_AladinParameters.symFlag = false;
  }
  Modified();
}


// ---------------------------------------------------------------------------
// OnAladinIterationsMaxSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnAladinIterationsMaxSpinBoxValueChanged(int value)
{
  m_RegParameters.m_AladinParameters.maxiterationNumber = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnPercentBlockSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnPercentBlockSpinBoxValueChanged(int value)
{
  m_RegParameters.m_AladinParameters.block_percent_to_use = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnPercentInliersSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnPercentInliersSpinBoxValueChanged(int value)
{
  m_RegParameters.m_AladinParameters.inlier_lts = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnAladinInterpolationNearestRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnAladinInterpolationNearestRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegParameters.m_AladinParameters.interpolation = NEAREST_INTERPOLATION;
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnAladinInterpolationLinearRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnAladinInterpolationLinearRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegParameters.m_AladinParameters.interpolation = LINEAR_INTERPOLATION;
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnAladinInterpolationCubicRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnAladinInterpolationCubicRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegParameters.m_AladinParameters.interpolation = CUBIC_INTERPOLATION;
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnNonRigidInputControlPointCheckBoxStateChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNonRigidInputControlPointCheckBoxStateChanged( int state )
{
  if ( state == Qt::Checked )
  {
    m_RegParameters.m_F3dParameters.inputControlPointGridFlag = true;
    m_Controls.m_NonRigidInputControlPointFileNameLineEdit->setEnabled( true );
  }
  else
  {
    m_RegParameters.m_F3dParameters.inputControlPointGridFlag = false;
    m_Controls.m_NonRigidInputControlPointFileNameLineEdit->setEnabled( false );
  }    

  Modified();
}


// ---------------------------------------------------------------------------
// OnNonRigidInputControlPointBrowsePushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNonRigidInputControlPointBrowsePushButtonPressed( void )
{
  m_RegParameters.m_F3dParameters.inputControlPointGridName = 
    QFileDialog::getOpenFileName(NULL, 
				 tr("Select spline control point nifti file"), 
				 QDir::currentPath(), 
				 tr("Control point file (*.nii);;"
				    "Any file (*)"));

  if ( ! m_RegParameters.m_F3dParameters.inputControlPointGridName.isEmpty() )
  {
    m_RegParameters.m_F3dParameters.inputControlPointGridFlag = true;
    m_Controls.m_NonRigidInputControlPointFileNameLineEdit->setText( m_RegParameters.
								     m_F3dParameters.
								     inputControlPointGridName );
  }
  else
  {
    m_RegParameters.m_F3dParameters.inputControlPointGridFlag = false;
    m_Controls.m_NonRigidInputControlPointFileNameLineEdit->setText( m_RegParameters.
								     m_F3dParameters.
								     inputControlPointGridName );
  }

  m_Controls.m_NonRigidInputControlPointCheckBox->setChecked( m_RegParameters.m_F3dParameters.
							      inputControlPointGridFlag );

  Modified();
}


// ---------------------------------------------------------------------------
// OnLowerThresholdTargetImageDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnLowerThresholdTargetImageDoubleSpinBoxValueChanged(double value)
{
 m_RegParameters.m_F3dParameters.referenceThresholdLow = value;

 if ( ( m_RegParameters.m_F3dParameters.referenceThresholdLow != -std::numeric_limits<PrecisionTYPE>::max() )
      && ( m_RegParameters.m_F3dParameters.referenceThresholdUp != -std::numeric_limits<PrecisionTYPE>::max() )
      && ( value > m_RegParameters.m_F3dParameters.referenceThresholdUp ) )
 {
   m_RegParameters.m_F3dParameters.referenceThresholdUp = -std::numeric_limits<PrecisionTYPE>::max();

   m_Controls.m_UpperThresholdTargetImageDoubleSpinBox
     ->setValue( m_RegParameters.m_F3dParameters.referenceThresholdUp );
 }

 Modified();
}


// ---------------------------------------------------------------------------
// OnLowerThresholdTargetImageAutoPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnLowerThresholdTargetImageAutoPushButtonPressed( void )
{
  m_RegParameters.m_F3dParameters.referenceThresholdLow = -std::numeric_limits<PrecisionTYPE>::max();

  m_Controls.m_LowerThresholdTargetImageDoubleSpinBox
    ->setValue( m_RegParameters.m_F3dParameters.referenceThresholdLow );

  Modified();
}


// ---------------------------------------------------------------------------
// OnUpperThresholdTargetImageDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnUpperThresholdTargetImageDoubleSpinBoxValueChanged(double value)
{
 m_RegParameters.m_F3dParameters.referenceThresholdUp = value;

 if ( ( m_RegParameters.m_F3dParameters.referenceThresholdUp != -std::numeric_limits<PrecisionTYPE>::max() ) 
      && ( m_RegParameters.m_F3dParameters.referenceThresholdLow != -std::numeric_limits<PrecisionTYPE>::max() ) 
      && ( value < m_RegParameters.m_F3dParameters.referenceThresholdLow ) )
 {
   m_RegParameters.m_F3dParameters.referenceThresholdLow = -std::numeric_limits<PrecisionTYPE>::max();

   m_Controls.m_LowerThresholdTargetImageDoubleSpinBox
     ->setValue( m_RegParameters.m_F3dParameters.referenceThresholdLow );
 }

 Modified();
}


// ---------------------------------------------------------------------------
// OnUpperThresholdTargetImageAutoPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnUpperThresholdTargetImageAutoPushButtonPressed( void )
{
  m_RegParameters.m_F3dParameters.referenceThresholdUp = -std::numeric_limits<PrecisionTYPE>::max();

  m_Controls.m_UpperThresholdTargetImageDoubleSpinBox
    ->setValue( m_RegParameters.m_F3dParameters.referenceThresholdUp );

  Modified();
}


// ---------------------------------------------------------------------------
// OnLowerThresholdSourceImageDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnLowerThresholdSourceImageDoubleSpinBoxValueChanged(double value)
{
  m_RegParameters.m_F3dParameters.floatingThresholdLow = value;

  if ( ( m_RegParameters.m_F3dParameters.floatingThresholdLow != -std::numeric_limits<PrecisionTYPE>::max() ) 
       && ( m_RegParameters.m_F3dParameters.floatingThresholdUp != -std::numeric_limits<PrecisionTYPE>::max() ) 
       && ( value > m_RegParameters.m_F3dParameters.floatingThresholdUp ) )
 {
   m_RegParameters.m_F3dParameters.floatingThresholdUp = -std::numeric_limits<PrecisionTYPE>::max();

   m_Controls.m_UpperThresholdSourceImageDoubleSpinBox
     ->setValue( m_RegParameters.m_F3dParameters.floatingThresholdUp );
 }

  Modified();
}


// ---------------------------------------------------------------------------
// OnLowerThresholdSourceImageAutoPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnLowerThresholdSourceImageAutoPushButtonPressed( void )
{
  m_RegParameters.m_F3dParameters.floatingThresholdLow = -std::numeric_limits<PrecisionTYPE>::max();

  m_Controls.m_LowerThresholdSourceImageDoubleSpinBox
    ->setValue( m_RegParameters.m_F3dParameters.floatingThresholdLow );

  Modified();
}


// ---------------------------------------------------------------------------
// OnUpperThresholdSourceImageDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnUpperThresholdSourceImageDoubleSpinBoxValueChanged(double value)
{
  m_RegParameters.m_F3dParameters.floatingThresholdUp = value;

  if ( ( m_RegParameters.m_F3dParameters.floatingThresholdUp != -std::numeric_limits<PrecisionTYPE>::max() ) 
       && ( m_RegParameters.m_F3dParameters.floatingThresholdLow != -std::numeric_limits<PrecisionTYPE>::max() ) 
       && ( value < m_RegParameters.m_F3dParameters.floatingThresholdLow ) )
 {
   m_RegParameters.m_F3dParameters.floatingThresholdLow = -std::numeric_limits<PrecisionTYPE>::max();

   m_Controls.m_LowerThresholdSourceImageDoubleSpinBox
     ->setValue( m_RegParameters.m_F3dParameters.floatingThresholdLow );
 }

  Modified();
}


// ---------------------------------------------------------------------------
// OnUpperThresholdSourceImageAutoPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnUpperThresholdSourceImageAutoPushButtonPressed( void )
{
  m_RegParameters.m_F3dParameters.floatingThresholdUp = -std::numeric_limits<PrecisionTYPE>::max();

  m_Controls.m_UpperThresholdSourceImageDoubleSpinBox
    ->setValue( m_RegParameters.m_F3dParameters.floatingThresholdUp );

  Modified();
}


// ---------------------------------------------------------------------------
// OnControlPointSpacingXDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnControlPointSpacingXDoubleSpinBoxValueChanged(double value)
{
  m_RegParameters.m_F3dParameters.spacing[0] = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnControlPointSpacingYDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnControlPointSpacingYDoubleSpinBoxValueChanged(double value)
{
  m_RegParameters.m_F3dParameters.spacing[1] = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnControlPointSpacingZDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnControlPointSpacingZDoubleSpinBoxValueChanged(double value)
{
  m_RegParameters.m_F3dParameters.spacing[2] = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnNumberSourceHistogramBinsSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNumberSourceHistogramBinsSpinBoxValueChanged(int value)
{
  m_RegParameters.m_F3dParameters.floatingBinNumber = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnNumberTargetHistogramBinsSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNumberTargetHistogramBinsSpinBoxValueChanged(int value)
{
  m_RegParameters.m_F3dParameters.referenceBinNumber = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnWeightBendingEnergyDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnWeightBendingEnergyDoubleSpinBoxValueChanged(double value)
{
  m_RegParameters.m_F3dParameters.bendingEnergyWeight = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnWeightLogJacobianDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnWeightLogJacobianDoubleSpinBoxValueChanged(double value)
{
  m_RegParameters.m_F3dParameters.jacobianLogWeight = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnLinearEnergyWeightsDoubleSpinBox_1ValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnLinearEnergyWeightsDoubleSpinBox_1ValueChanged(double value)
{
  m_RegParameters.m_F3dParameters.linearEnergyWeight0 = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnLinearEnergyWeightsDoubleSpinBox_2ValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnLinearEnergyWeightsDoubleSpinBox_2ValueChanged(double value)
{
  m_RegParameters.m_F3dParameters.linearEnergyWeight1 = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnApproxJacobianLogCheckBoxStateChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnApproxJacobianLogCheckBoxStateChanged( int state )
{
  if ( state == Qt::Checked )
  {
    m_RegParameters.m_F3dParameters.jacobianLogApproximation = true;
  }
  else
  {
    m_RegParameters.m_F3dParameters.jacobianLogApproximation = false;
  }    
  Modified();
}


// ---------------------------------------------------------------------------
// OnSimilarityNMIRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnSimilarityNMIRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegParameters.m_F3dParameters.similarity = NMI_SIMILARITY;
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnSimilaritySSDRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnSimilaritySSDRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegParameters.m_F3dParameters.similarity = SSD_SIMILARITY;
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnSimilarityKLDivRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnSimilarityKLDivRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegParameters.m_F3dParameters.similarity = KLDIV_SIMILARITY;
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnUseSimpleGradientAscentCheckBoxStateChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnUseSimpleGradientAscentCheckBoxStateChanged( int state )
{
  if ( state == Qt::Checked )
  {
    m_RegParameters.m_F3dParameters.useConjugate = false;
  }
  else
  {
    m_RegParameters.m_F3dParameters.useConjugate = true;
  }    
  Modified();
}


// ---------------------------------------------------------------------------
// OnUsePyramidalCheckBoxStateChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnUsePyramidalCheckBoxStateChanged( int state )
{
  if ( state == Qt::Checked )
  {
    m_RegParameters.m_F3dParameters.noPyramid = false;
  }
  else
  {
    m_RegParameters.m_F3dParameters.noPyramid = true;
  }    
  Modified();
}


// ---------------------------------------------------------------------------
// OnNonRigidIterationsMaxSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNonRigidIterationsMaxSpinBoxValueChanged(int value)
{
  m_RegParameters.m_F3dParameters.maxiterationNumber = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnSmoothingMetricDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnSmoothingMetricDoubleSpinBoxValueChanged(double value)
{
  m_RegParameters.m_F3dParameters.gradientSmoothingSigma = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnWarpedPaddingValueDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnWarpedPaddingValueDoubleSpinBoxValueChanged(double value)
{
  m_RegParameters.m_F3dParameters.warpedPaddingValue = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnWarpedPaddingValuePushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnWarpedPaddingValuePushButtonPressed( void )
{
  m_RegParameters.m_F3dParameters.warpedPaddingValue =  -std::numeric_limits<PrecisionTYPE>::max();
  m_Controls.m_WarpedPaddingValueDoubleSpinBox->setValue( m_RegParameters.m_F3dParameters.warpedPaddingValue );

  Modified();
}


// ---------------------------------------------------------------------------
// OnNonRigidNearestInterpolationRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNonRigidNearestInterpolationRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegParameters.m_F3dParameters.interpolation = NEAREST_INTERPOLATION;
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnNonRigidLinearInterpolationRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNonRigidLinearInterpolationRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegParameters.m_F3dParameters.interpolation = LINEAR_INTERPOLATION;
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnNonRigidCubicInterpolationRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNonRigidCubicInterpolationRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegParameters.m_F3dParameters.interpolation = CUBIC_INTERPOLATION;
    Modified();
  }
}


