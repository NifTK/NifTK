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

// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>

// Qmitk
#include "QmitkNiftyRegView.h"

// Qt
#include <QMessageBox>
#include <QFileDialog>

#include "mitkDataStorageUtils.h"
#include "mitkNodePredicateDataType.h"
#include "mitkImageToNifti.h"
#include "niftiImageToMitk.h"

#include "RegistrationExecution.h"

const std::string QmitkNiftyRegView::VIEW_ID = "uk.ac.ucl.cmic.views.niftyregview";

#define USE_QT_TREADING

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

QmitkNiftyRegView::QmitkNiftyRegView()
{

  m_ReferenceImage = 0;
  m_FloatingImage = 0;
  m_ReferenceMaskImage = 0;
  m_ControlPointGridImage = 0;

  SetDefaultParameters();

}


// ---------------------------------------------------------------------------
// DeallocateImages();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::DeallocateImages( void )
{
  if ( m_ReferenceImage ) 
  {
    nifti_image_free( m_ReferenceImage );
    m_ReferenceImage = 0;
  }

  if ( m_FloatingImage ) 
  {
    nifti_image_free( m_FloatingImage );
    m_FloatingImage = 0;
  }
    
  if ( m_ReferenceMaskImage )
  {
    nifti_image_free( m_ReferenceMaskImage );
    m_ReferenceMaskImage = 0;
  }
    
  if ( m_ControlPointGridImage )
  {
    nifti_image_free( m_ControlPointGridImage );
    m_ControlPointGridImage = 0;
  }
}


// ---------------------------------------------------------------------------
// SetDefaultParameters()
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::SetDefaultParameters()
{

  m_Modified = true;


  // Multi-Scale Options
    
  m_LevelNumber = 3;		// Number of level to perform
  m_Level2Perform = 3;		// Only perform the first levels 

  // Input Image Options

  m_TargetSigmaValue = 0;  // Smooth the target image using the specified sigma (mm) 
  m_SourceSigmaValue = 0;  // Smooth the source image using the specified sigma (mm)

  // Flag indicating whether to do rigid and/or non-rigid registrations

  m_FlagDoInitialRigidReg = true;
  m_FlagDoNonRigidReg = true;


  // Initial affine transformation
 
  m_FlagInputAffine = false;
  m_FlagFlirtAffine = false;

  m_InputAffineName.clear();

  
  // Progress bar parameters
  m_ProgressBarOffset = 0.;
  m_ProgressBarRange = 100.;


  // Initialise the 'reg_aladin' parameters
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  m_RegAladinParameters.outputResultFlag = false;
  m_RegAladinParameters.outputResultName.clear();

  m_RegAladinParameters.outputAffineFlag = false;
  m_RegAladinParameters.outputAffineName.clear();

  // Aladin - Initialisation

  m_RegAladinParameters.alignCenterFlag = true;    
  
  // Aladin - Method

  m_RegAladinParameters.regnType = RIGID_THEN_AFFINE;    

  m_RegAladinParameters.maxiterationNumber = 5;

  m_RegAladinParameters.block_percent_to_use = 50;
  m_RegAladinParameters.inlier_lts = 50;

  // Aladin - Advanced

  m_RegAladinParameters.interpolation = LINEAR_INTERPOLATION;


  // Initialise the 'reg_f3d' parameters
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Non-Rigid - Initialisation
 
  m_RegF3dParameters.inputControlPointGridFlag = false;
  m_RegF3dParameters.inputControlPointGridName.clear();

  // Non-Rigid - Output Options
 
  m_RegF3dParameters.outputControlPointGridName.clear();
  m_RegF3dParameters.outputWarpedName.clear();

  // Non-Rigid - Input Image

  m_RegF3dParameters.referenceThresholdUp  = -std::numeric_limits<PrecisionTYPE>::max();
  m_RegF3dParameters.referenceThresholdLow = -std::numeric_limits<PrecisionTYPE>::max();

  m_RegF3dParameters.floatingThresholdUp   = -std::numeric_limits<PrecisionTYPE>::max();
  m_RegF3dParameters.floatingThresholdLow  = -std::numeric_limits<PrecisionTYPE>::max();

  // Non-Rigid - Spline

  m_RegF3dParameters.spacing[0] = -5.;
  m_RegF3dParameters.spacing[1] = -5.;
  m_RegF3dParameters.spacing[2] = -5.;

  // Non-Rigid - Objective Function
 
  m_RegF3dParameters.referenceBinNumber = 64;
  m_RegF3dParameters.floatingBinNumber  = 64;

  m_RegF3dParameters.bendingEnergyWeight = 0.005;

  m_RegF3dParameters.linearEnergyWeight0 = 0.;
  m_RegF3dParameters.linearEnergyWeight1 = 0.;

  m_RegF3dParameters.jacobianLogWeight = 0.;

  m_RegF3dParameters.jacobianLogApproximation = true;

  m_RegF3dParameters.similarity = NMI_SIMILARITY;

  // Non-Rigid - Optimisation
 
  m_RegF3dParameters.useConjugate = true;
  m_RegF3dParameters.maxiterationNumber = 300;
  m_RegF3dParameters.noPyramid = false;

  // Non-Rigid - GPU-related options:
  
  m_RegF3dParameters.checkMem = false;
  m_RegF3dParameters.useGPU = false;
  m_RegF3dParameters.cardNumber = -1;

  // Non-Rigid - Advanced

  m_RegF3dParameters.interpolation = LINEAR_INTERPOLATION;

  m_RegF3dParameters.gradientSmoothingSigma = 0.;
  m_RegF3dParameters.warpedPaddingValue = -std::numeric_limits<PrecisionTYPE>::max();
  m_RegF3dParameters.verbose = true;

}


// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

QmitkNiftyRegView::~QmitkNiftyRegView()
{
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
  m_Controls.m_NumberOfLevelsSpinBox->setValue( m_LevelNumber );

  m_Controls.m_LevelsToPerformSpinBox->setKeyboardTracking ( false );
  m_Controls.m_LevelsToPerformSpinBox->setMaximum( 10 );
  m_Controls.m_LevelsToPerformSpinBox->setMinimum( 1 );
  m_Controls.m_LevelsToPerformSpinBox->setValue( m_Level2Perform );

  // Input Image Options
 
  m_Controls.m_SmoothSourceImageDoubleSpinBox->setKeyboardTracking ( false );
  m_Controls.m_SmoothSourceImageDoubleSpinBox->setMaximum( 1000 );
  m_Controls.m_SmoothSourceImageDoubleSpinBox->setMinimum( 0 );
  m_Controls.m_SmoothSourceImageDoubleSpinBox->setValue( m_SourceSigmaValue );

  m_Controls.m_SmoothTargetImageDoubleSpinBox->setKeyboardTracking ( false );
  m_Controls.m_SmoothTargetImageDoubleSpinBox->setMaximum( 1000 );
  m_Controls.m_SmoothTargetImageDoubleSpinBox->setMinimum( 0 );
  m_Controls.m_SmoothTargetImageDoubleSpinBox->setValue( m_TargetSigmaValue );

  m_Controls.m_DoBlockMatchingOnlyRadioButton->setChecked( m_FlagDoInitialRigidReg 
							   && ( ! m_FlagDoNonRigidReg ) );
  m_Controls.m_DoNonRigidOnlyRadioButton->setChecked( ( ! m_FlagDoInitialRigidReg )
						      && m_FlagDoNonRigidReg );
  m_Controls.m_DoBlockMatchingThenNonRigidRadioButton->setChecked( m_FlagDoInitialRigidReg 
								   && m_FlagDoNonRigidReg );


  // Initial Affine Transformation

  m_Controls.m_InitialAffineTransformationGroupBox->setChecked( m_FlagInputAffine );
  m_Controls.m_InputFlirtCheckBox->setChecked( m_FlagFlirtAffine );


  // Aladin Parameters
  // ~~~~~~~~~~~~~~~~~

  // Aladin - Initialisation

  m_Controls.m_UseNiftyHeaderCheckBox->setChecked( m_RegAladinParameters.alignCenterFlag );

  // Aladin - Method

  m_Controls.m_RigidOnlyRadioButton->setChecked( m_RegAladinParameters.regnType 
						 == RIGID_ONLY );
  m_Controls.m_RigidThenAffineRadioButton->setChecked( m_RegAladinParameters.regnType 
						       == RIGID_THEN_AFFINE );
  m_Controls.m_DirectAffineRadioButton->setChecked( m_RegAladinParameters.regnType 
						    == DIRECT_AFFINE );

  m_Controls.m_AladinIterationsMaxSpinBox->setKeyboardTracking ( false );
  m_Controls.m_AladinIterationsMaxSpinBox->setMaximum( 1000 );
  m_Controls.m_AladinIterationsMaxSpinBox->setMinimum( 1 );
  m_Controls.m_AladinIterationsMaxSpinBox->setValue( m_RegAladinParameters.maxiterationNumber );

  m_Controls.m_PercentBlockSpinBox->setKeyboardTracking ( false );
  m_Controls.m_PercentBlockSpinBox->setMaximum( 100 );
  m_Controls.m_PercentBlockSpinBox->setMinimum( 1 );
  m_Controls.m_PercentBlockSpinBox->setValue( m_RegAladinParameters.block_percent_to_use );

  m_Controls.m_PercentInliersSpinBox->setKeyboardTracking ( false );
  m_Controls.m_PercentInliersSpinBox->setMaximum( 100 );
  m_Controls.m_PercentInliersSpinBox->setMinimum( 1 );
  m_Controls.m_PercentInliersSpinBox->setValue( m_RegAladinParameters.inlier_lts );

  // Aladin - Advanced

  m_Controls.m_AladinInterpolationNearestRadioButton->setChecked( m_RegAladinParameters
								  .interpolation 
								  == NEAREST_INTERPOLATION );
  m_Controls.m_AladinInterpolationLinearRadioButton->setChecked( m_RegAladinParameters
								  .interpolation 
								  == LINEAR_INTERPOLATION );
  m_Controls.m_AladinInterpolationCubicRadioButton->setChecked( m_RegAladinParameters
								.interpolation 
								== CUBIC_INTERPOLATION );

  // Non-Rigid Parameters
  // ~~~~~~~~~~~~~~~~~~~~
  
  // Non-Rigid - Initialisation

  m_Controls.m_NonRigidInputControlPointCheckBox->setChecked( m_RegF3dParameters.
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
    ->setValue( m_RegF3dParameters.referenceThresholdLow );

  // UpperThresholdTargetImageCheckBox
  m_Controls.m_UpperThresholdTargetImageDoubleSpinBox->setKeyboardTracking ( false );

  m_Controls.m_UpperThresholdTargetImageDoubleSpinBox
    ->setMaximum( std::numeric_limits<PrecisionTYPE>::max() );
  m_Controls.m_UpperThresholdTargetImageDoubleSpinBox
    ->setMinimum( -std::numeric_limits<PrecisionTYPE>::max() );

  m_Controls.m_UpperThresholdTargetImageDoubleSpinBox->setSpecialValueText(tr("max"));

  m_Controls.m_UpperThresholdTargetImageDoubleSpinBox
    ->setValue( m_RegF3dParameters.referenceThresholdUp );

  // LowerThresholdSourceImage
  m_Controls.m_LowerThresholdSourceImageDoubleSpinBox->setKeyboardTracking ( false );

  m_Controls.m_LowerThresholdSourceImageDoubleSpinBox
    ->setMaximum( std::numeric_limits<PrecisionTYPE>::max() );
  m_Controls.m_LowerThresholdSourceImageDoubleSpinBox
    ->setMinimum( -std::numeric_limits<PrecisionTYPE>::max() );

  m_Controls.m_LowerThresholdSourceImageDoubleSpinBox->setSpecialValueText(tr("min"));

  m_Controls.m_LowerThresholdSourceImageDoubleSpinBox
    ->setValue( m_RegF3dParameters.floatingThresholdLow );

  // UpperThresholdSourceImage
  m_Controls.m_UpperThresholdSourceImageDoubleSpinBox->setKeyboardTracking ( false );

  m_Controls.m_UpperThresholdSourceImageDoubleSpinBox
    ->setMaximum( std::numeric_limits<PrecisionTYPE>::max() );
  m_Controls.m_UpperThresholdSourceImageDoubleSpinBox
    ->setMinimum( -std::numeric_limits<PrecisionTYPE>::max() );

  m_Controls.m_UpperThresholdSourceImageDoubleSpinBox->setSpecialValueText(tr("max"));

  m_Controls.m_UpperThresholdSourceImageDoubleSpinBox
    ->setValue( m_RegF3dParameters.floatingThresholdUp );

  // Non-Rigid - Spline

  m_Controls.m_ControlPointSpacingXDoubleSpinBox->setMinimum( -50 );
  m_Controls.m_ControlPointSpacingXDoubleSpinBox->setMaximum(  50 );
  m_Controls.m_ControlPointSpacingXDoubleSpinBox->setValue( m_RegF3dParameters.spacing[0] );

  m_Controls.m_ControlPointSpacingYDoubleSpinBox->setMinimum( -50 );
  m_Controls.m_ControlPointSpacingYDoubleSpinBox->setMaximum(  50 );
  m_Controls.m_ControlPointSpacingYDoubleSpinBox->setValue( m_RegF3dParameters.spacing[1] );

  m_Controls.m_ControlPointSpacingZDoubleSpinBox->setMinimum( -50 );
  m_Controls.m_ControlPointSpacingZDoubleSpinBox->setMaximum(  50 );
  m_Controls.m_ControlPointSpacingZDoubleSpinBox->setValue( m_RegF3dParameters.spacing[2] );

  // Non-Rigid - Objective Function
    
  m_Controls.m_NumberSourceHistogramBinsSpinBox->setMinimum( 4 );
  m_Controls.m_NumberSourceHistogramBinsSpinBox->setMaximum( 256 );
  m_Controls.m_NumberSourceHistogramBinsSpinBox->setValue( m_RegF3dParameters.
							   referenceBinNumber );

  m_Controls.m_NumberTargetHistogramBinsSpinBox->setMinimum( 4 );
  m_Controls.m_NumberTargetHistogramBinsSpinBox->setMaximum( 256 );
  m_Controls.m_NumberTargetHistogramBinsSpinBox->setValue( m_RegF3dParameters.
							   floatingBinNumber );

  m_Controls.m_WeightBendingEnergyDoubleSpinBox->setMinimum( 0 );
  m_Controls.m_WeightBendingEnergyDoubleSpinBox->setMaximum( 1 );
  m_Controls.m_WeightBendingEnergyDoubleSpinBox->setValue( m_RegF3dParameters.
							   bendingEnergyWeight );

  m_Controls.m_WeightLogJacobianDoubleSpinBox->setMinimum( 0 );
  m_Controls.m_WeightLogJacobianDoubleSpinBox->setMaximum( 1 );
  m_Controls.m_WeightLogJacobianDoubleSpinBox->setValue( m_RegF3dParameters.
							 jacobianLogWeight );

  m_Controls.m_LinearEnergyWeightsDoubleSpinBox_1->setMinimum( 0 );
  m_Controls.m_LinearEnergyWeightsDoubleSpinBox_1->setMaximum( 100 );
  m_Controls.m_LinearEnergyWeightsDoubleSpinBox_1->setValue( m_RegF3dParameters.
							     linearEnergyWeight0 );

  m_Controls.m_LinearEnergyWeightsDoubleSpinBox_2->setMinimum( 0 );
  m_Controls.m_LinearEnergyWeightsDoubleSpinBox_2->setMaximum( 100 );
  m_Controls.m_LinearEnergyWeightsDoubleSpinBox_2->setValue( m_RegF3dParameters.
							     linearEnergyWeight1 );

  m_Controls.m_ApproxJacobianLogCheckBox->setChecked( m_RegF3dParameters.
						      jacobianLogApproximation );

  m_Controls.m_SimilarityNMIRadioButton->setChecked( m_RegF3dParameters.similarity 
						     == NMI_SIMILARITY );
  m_Controls.m_SimilaritySSDRadioButton->setChecked( m_RegF3dParameters.similarity 
						     == SSD_SIMILARITY );
  m_Controls.m_SimilarityKLDivRadioButton->setChecked( m_RegF3dParameters.similarity 
						       == KLDIV_SIMILARITY );

  // Non-Rigid - Optimisation

  m_Controls.m_UseSimpleGradientAscentCheckBox->setChecked( ! m_RegF3dParameters.useConjugate );

  m_Controls.m_NonRigidIterationsMaxSpinBox->setMaximum( 100 );
  m_Controls.m_NonRigidIterationsMaxSpinBox->setMinimum( 1 );
  m_Controls.m_NonRigidIterationsMaxSpinBox->setValue( m_RegF3dParameters.maxiterationNumber );

  m_Controls.m_UsePyramidalCheckBox->setChecked( ! m_RegF3dParameters.noPyramid );

  // Non-Rigid - Advanced

  m_Controls.m_SmoothingMetricDoubleSpinBox->setMaximum( 50 );
  m_Controls.m_SmoothingMetricDoubleSpinBox->setMinimum( 0 );
  m_Controls.m_SmoothingMetricDoubleSpinBox->setValue( m_RegF3dParameters.gradientSmoothingSigma );

  m_Controls.m_WarpedPaddingValueDoubleSpinBox
    ->setMaximum( std::numeric_limits<PrecisionTYPE>::max() );
  m_Controls.m_WarpedPaddingValueDoubleSpinBox
    ->setMinimum( -std::numeric_limits<PrecisionTYPE>::max() );
  m_Controls.m_WarpedPaddingValueDoubleSpinBox->setSpecialValueText(tr("none"));
  m_Controls.m_WarpedPaddingValueDoubleSpinBox->setValue( m_RegF3dParameters.warpedPaddingValue );


  m_Controls.m_NonRigidNearestInterpolationRadioButton->setChecked( m_RegF3dParameters.
								    interpolation 
								   == NEAREST_INTERPOLATION );

  m_Controls.m_NonRigidLinearInterpolationRadioButton->setChecked( m_RegF3dParameters.
								   interpolation 
								   == LINEAR_INTERPOLATION );

  m_Controls.m_NonRigidCubicInterpolationRadioButton->setChecked( m_RegF3dParameters.
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

  connect( m_Controls.m_CancelPushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnCancelPushButtonPressed( void ) ) );

  connect( m_Controls.m_ResetParametersPushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnResetParametersPushButtonPressed( void ) ) );

  connect( m_Controls.m_SaveAsPushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnSaveAsPushButtonPressed( void ) ) );

  connect( m_Controls.m_ExecutePushButton,
	   SIGNAL( pressed( void ) ),
	   this,
	   SLOT( OnExecutePushButtonPressed( void ) ) );
}


// ---------------------------------------------------------------------------
// PrintSelf
// ---------------------------------------------------------------------------

void QmitkNiftyRegView::PrintSelf( std::ostream& os )
{

  // Initial affine transformation
 
  os << "# Initial affine transformation" << std::endl;

  if ( m_InputAffineName.isEmpty() )
    os << "InputAffineName: UNSET" << std::endl;
  else
    os << "InputAffineName: " << m_InputAffineName.toStdString() << std::endl;

  os << "InputAffineFlag: " << m_FlagInputAffine << std::endl;
  os << "FlirtAffineFlag: " << m_FlagFlirtAffine << std::endl;
  		               

  // ---------------------------------------------------------------------------
  // Rigid/Affine Aladin Parameters
  // ---------------------------------------------------------------------------

  os << "# Rigid/Affine Aladin Parameters" << std::endl;

  os << "Aladin-outputResultFlag: " << m_RegAladinParameters.outputResultFlag << std::endl;

  if ( m_RegAladinParameters.outputResultName.isEmpty() )
    os << "Aladin-outputResultName: UNSET" << std::endl;
  else
    os << "Aladin-outputResultName: " << m_RegAladinParameters.outputResultName.toStdString() << std::endl;
	                       
  os << "Aladin-outputAffineFlag: " << m_RegAladinParameters.outputAffineFlag << std::endl;

  if ( m_RegAladinParameters.outputAffineName.isEmpty() )
    os << "Aladin-outputAffineName: UNSET" << std::endl;
  else
    os << "Aladin-outputAffineName: " << m_RegAladinParameters.outputAffineName.toStdString() << std::endl;

  // Aladin - Initialisation

  os << "# Aladin - Initialisation" << std::endl;

  os << "Aladin-alignCenterFlag: " << m_RegAladinParameters.alignCenterFlag << std::endl;
  
  // Aladin - Method
  
  os << "# Aladin - Method" << std::endl;

  os << "Aladin-regnType: " << m_RegAladinParameters.regnType << std::endl;
  
  os << "Aladin-maxiterationNumber: " << m_RegAladinParameters.maxiterationNumber << std::endl;
  
  os << "Aladin-block_percent_to_use: " << m_RegAladinParameters.block_percent_to_use << std::endl;
  os << "Aladin-inlier_lts: " << m_RegAladinParameters.inlier_lts << std::endl;
  
  // Aladin - Advanced
  
  os << "# Aladin - Advanced" << std::endl;

  os << "Aladin-interpolation: " << m_RegAladinParameters.interpolation << std::endl;
  

  // ---------------------------------------------------------------------------
  // Non-rigid Parameters
  // ---------------------------------------------------------------------------

  // Initial transformation options (One option will be considered):
 
  os << "# Non-Rigid, F3D - Initial transformation options" << std::endl;

  os << "F3D-inputControlPointGridFlag: " << m_RegF3dParameters.inputControlPointGridFlag << std::endl;

  if ( m_RegF3dParameters.inputControlPointGridName.isEmpty() )
    os << "F3D-inputControlPointGridName: UNSET" << std::endl;
  else
    os << "F3D-inputControlPointGridName: " << m_RegF3dParameters.inputControlPointGridName.toStdString() << std::endl;

  // Output options:
 
  os << "# Non-Rigid, F3D - Output options" << std::endl;

  if ( m_RegF3dParameters.outputControlPointGridName.isEmpty() )
    os << "F3D-outputControlPointGridName: UNSET" << std::endl;
  else
    os << "F3D-outputControlPointGridName: " << m_RegF3dParameters.outputControlPointGridName.toStdString() << std::endl;

  if ( m_RegF3dParameters.outputWarpedName.isEmpty() )
    os << "F3D-outputWarpedName: UNSET" << std::endl;
  else
    os << "F3D-outputWarpedName: " << m_RegF3dParameters.outputWarpedName.toStdString() << std::endl;	    

  // Input image options:

  os << "# Non-Rigid, F3D - Input image options" << std::endl;

  if ( m_RegF3dParameters.referenceThresholdUp == -std::numeric_limits<PrecisionTYPE>::max() )
    os << "F3D-referenceThresholdUp: max" << std::endl;    
  else
    os << "F3D-referenceThresholdUp: " << m_RegF3dParameters.referenceThresholdUp << std::endl; 

  if ( m_RegF3dParameters.referenceThresholdLow == -std::numeric_limits<PrecisionTYPE>::max() )
    os << "F3D-referenceThresholdLow: min" << std::endl;    
  else
    os << "F3D-referenceThresholdLow: " << m_RegF3dParameters.referenceThresholdLow << std::endl; 

  if ( m_RegF3dParameters.floatingThresholdUp == -std::numeric_limits<PrecisionTYPE>::max() )
    os << "F3D-floatingThresholdUp: max" << std::endl;    
  else
    os << "F3D-floatingThresholdUp: " << m_RegF3dParameters.floatingThresholdUp << std::endl;  

  if ( m_RegF3dParameters.floatingThresholdLow == -std::numeric_limits<PrecisionTYPE>::max() )
    os << "F3D-floatingThresholdLow: min" << std::endl;    
  else
    os << "F3D-floatingThresholdLow: " << m_RegF3dParameters.floatingThresholdLow << std::endl; 

  // Spline options:
 
  os << "# Non-Rigid, F3D - Spline options" << std::endl;

  os << "F3D-spacing: " 
     << m_RegF3dParameters.spacing[0] << " "
     << m_RegF3dParameters.spacing[1] << " "
     << m_RegF3dParameters.spacing[2]
     << std::endl;

  // Objective function options:
 
  os << "# Non-Rigid, F3D - Objective function options" << std::endl;

  os << "F3D-referenceBinNumber: " << m_RegF3dParameters.referenceBinNumber << std::endl;
  os << "F3D-floatingBinNumber: " << m_RegF3dParameters.floatingBinNumber << std::endl; 

  os << "F3D-bendingEnergyWeight: " << m_RegF3dParameters.bendingEnergyWeight << std::endl;
	                        
  os << "F3D-linearEnergyWeight0: " << m_RegF3dParameters.linearEnergyWeight0 << std::endl;
  os << "F3D-linearEnergyWeight1: " << m_RegF3dParameters.linearEnergyWeight1 << std::endl;

  os << "F3D-jacobianLogWeight: " << m_RegF3dParameters.jacobianLogWeight << std::endl;  

  os << "F3D-jacobianLogApproximation: " << m_RegF3dParameters.jacobianLogApproximation << std::endl;

  os << "F3D-similarity: " << m_RegF3dParameters.similarity << std::endl;

  // Optimisation options:
 
  os << "# Non-Rigid, F3D - Optimisation options" << std::endl;

  os << "F3D-useConjugate: " << m_RegF3dParameters.useConjugate << std::endl;      
  os << "F3D-maxiterationNumber: " << m_RegF3dParameters.maxiterationNumber << std::endl;
  os << "F3D-noPyramid: " << m_RegF3dParameters.noPyramid << std::endl; 

  // GPU-related options:

  os << "# Non-Rigid, F3D - GPU-related options" << std::endl;

  os << "F3D-checkMem: " << m_RegF3dParameters.checkMem << std::endl;  
  os << "F3D-useGPU: " << m_RegF3dParameters.useGPU << std::endl;    
  os << "F3D-cardNumber: " << m_RegF3dParameters.cardNumber << std::endl;

  // Other options:
  
  os << "# Non-Rigid, F3D - Other options" << std::endl;

  os << "F3D-interpolation: " << m_RegF3dParameters.interpolation << std::endl;

  os << "F3D-gradientSmoothingSigma: " << m_RegF3dParameters.gradientSmoothingSigma << std::endl;

  if ( m_RegF3dParameters.warpedPaddingValue == -std::numeric_limits<PrecisionTYPE>::max() )
    os << "F3D-warpedPaddingValue: auto" << std::endl;    
  else
    os << "F3D-warpedPaddingValue: " << m_RegF3dParameters.warpedPaddingValue << std::endl;    

  os << "F3D-verbose: " << m_RegF3dParameters.verbose << std::endl;               

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
// OnSourceImageComboBoxChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnSourceImageComboBoxChanged(int index)
{
  QString sourceName = m_Controls.m_SourceImageComboBox->currentText();
  QString targetName = m_Controls.m_TargetImageComboBox->currentText();

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
  Modified();
}


// ---------------------------------------------------------------------------
// OnTargetImageComboBoxChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnTargetImageComboBoxChanged(int index)
{
  QString sourceName = m_Controls.m_SourceImageComboBox->currentText();
  QString targetName = m_Controls.m_TargetImageComboBox->currentText();

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
// OnSaveAsPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnSaveAsPushButtonPressed( void )
{
  PrintSelf( std::cout );
}


// ---------------------------------------------------------------------------
// OnExecutePushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnExecutePushButtonPressed( void )
{

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
#ifdef USE_QT_TREADING

  QEventLoop q;
  RegistrationExecution regExecutionThread( this );

  // The cancel button terminates the registration
  connect( m_Controls.m_CancelPushButton, SIGNAL( pressed( ) ), 
	   &regExecutionThread, SLOT( quit( ) ) );

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

  if ( userData->m_FlagDoInitialRigidReg && userData->m_FlagDoNonRigidReg ) 
    userData->m_ProgressBarRange = 50.;
  else
    userData->m_ProgressBarRange = 100.;

  userData->m_ProgressBarOffset = 0.;


  // Create and run the Aladin registration?

  if ( userData->m_FlagDoInitialRigidReg ) 
  {
    reg_aladin<PrecisionTYPE> *regAladin;

    regAladin = userData->CreateAladinRegistrationObject( mitkSourceImage, 
							  mitkTargetImage, 
							  mitkTargetMaskImage );
  
    regAladin->SetProgressCallbackFunction( &UpdateProgressBar, userData );

    regAladin->Run();

    mitkSourceImage = ConvertNiftiImageToMitk( regAladin->GetFinalWarpedImage() );

    // Add this result to the data manager
    mitk::DataNode::Pointer resultNode = mitk::DataNode::New();

    std::string nameOfResultImage;
    if ( userData->m_RegAladinParameters.regnType == QmitkNiftyRegView::RIGID_ONLY )
      nameOfResultImage = "rigid registration to ";
    else
      nameOfResultImage = "affine registration to ";
    nameOfResultImage.append( nodeTarget->GetName() );

    resultNode->SetProperty("name", mitk::StringProperty::New(nameOfResultImage) );
    resultNode->SetData( mitkSourceImage );

    userData->GetDataStorage()->Add( resultNode, nodeSource );

    // Deallocate data
    userData->DeallocateImages();
    delete regAladin;

    UpdateProgressBar( 100., userData );

    if ( userData->m_FlagDoNonRigidReg ) 
      userData->m_ProgressBarOffset = 50.;
  }


  // Create and run the F3D registration

  if ( userData->m_FlagDoNonRigidReg ) 
  {
    reg_f3d<PrecisionTYPE> *regNonRigid;

    regNonRigid = userData->CreateNonRigidRegistrationObject( mitkSourceImage, 
							      mitkTargetImage, 
							      mitkTargetMaskImage );  

    regNonRigid->SetProgressCallbackFunction( &UpdateProgressBar, userData );

    regNonRigid->Run_f3d();

    mitkTransformedImage = ConvertNiftiImageToMitk( regNonRigid->GetWarpedImage()[0] );

    // Add this result to the data manager
    mitk::DataNode::Pointer resultNode = mitk::DataNode::New();

    std::string nameOfResultImage( "non-rigid registration to " );
    nameOfResultImage.append( nodeTarget->GetName() );

    resultNode->SetProperty("name", mitk::StringProperty::New(nameOfResultImage) );
    resultNode->SetData( mitkTransformedImage );

    userData->GetDataStorage()->Add( resultNode, nodeSource );

    // Deallocate data
    userData->DeallocateImages();
    delete regNonRigid;

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
// CreateAladinRegistrationObject();
// --------------------------------------------------------------------------- 

reg_aladin<PrecisionTYPE> *QmitkNiftyRegView::CreateAladinRegistrationObject( mitk::Image *mitkSourceImage, 
									      mitk::Image *mitkTargetImage, 
									      mitk::Image *mitkTargetMaskImage )
{
  reg_aladin<PrecisionTYPE> *REG = new reg_aladin<PrecisionTYPE>;
  
  // Get nifti versions of the images

  if ( m_ReferenceImage ) nifti_image_free( m_ReferenceImage );
  m_ReferenceImage = ConvertMitkImageToNifti( mitkTargetImage );

  if ( m_FloatingImage ) nifti_image_free( m_FloatingImage );
  m_FloatingImage  = ConvertMitkImageToNifti( mitkSourceImage );

#if 0
  nifti_set_filenames( m_ReferenceImage,"aladinReference.nii",0,0 );
  nifti_image_write( m_ReferenceImage );

  nifti_set_filenames( m_FloatingImage,"aladinFloating.nii",0,0 );
  nifti_image_write( m_FloatingImage );
#endif

  // Check the dimensions of the images

  reg_checkAndCorrectDimension( m_ReferenceImage );
  reg_checkAndCorrectDimension( m_FloatingImage );

  // Set the reference and floating image

  REG->SetInputReference( m_ReferenceImage );
  REG->SetInputFloating( m_FloatingImage );

  // Set the reference mask image 

  if ( mitkTargetMaskImage ) 
  {

    if ( m_ReferenceMaskImage ) nifti_image_free( m_ReferenceMaskImage );
    m_ReferenceMaskImage = ConvertMitkImageToNifti( mitkTargetMaskImage );

    reg_checkAndCorrectDimension(m_ReferenceMaskImage);

    // check the dimensions

    for ( int i=1; i<=m_ReferenceImage->dim[0]; i++ ) {
    
      if ( m_ReferenceImage->dim[i] != m_ReferenceMaskImage->dim[i] ) 
      {
	fprintf(stderr,"* ERROR The reference image and its mask do not have the same dimension\n");
	return 0;
      }
    }
    
    REG->SetInputMask( m_ReferenceMaskImage );
  }

  // Aladin - Initialisation
  
  REG->SetNumberOfLevels( m_LevelNumber );
  REG->SetLevelsToPerform( m_Level2Perform );
  
  REG->SetReferenceSigma( m_TargetSigmaValue );
  REG->SetFloatingSigma( m_SourceSigmaValue );
  
  if ( m_FlagInputAffine 
       && ( ! m_InputAffineName.isEmpty() ) )
    
    REG->SetInputTransform( strdup( m_InputAffineName.toStdString().c_str() ), 
			    m_FlagFlirtAffine );
  
  REG->SetAlignCentre( m_RegAladinParameters.alignCenterFlag );

  // Aladin - Method

  REG->SetPerformAffine( ( m_RegAladinParameters.regnType == RIGID_THEN_AFFINE )
			 || ( m_RegAladinParameters.regnType == DIRECT_AFFINE ) );

  REG->SetPerformRigid( ( m_RegAladinParameters.regnType == RIGID_ONLY )
			|| ( m_RegAladinParameters.regnType == RIGID_THEN_AFFINE ) );

  REG->SetMaxIterations( m_RegAladinParameters.maxiterationNumber );

  REG->SetBlockPercentage( m_RegAladinParameters.block_percent_to_use );
  REG->SetInlierLts( m_RegAladinParameters.inlier_lts );

  // Aladin - Advanced

  REG->SetInterpolation( m_RegAladinParameters.interpolation );

  return REG;
}


// ---------------------------------------------------------------------------
// CreateNonRigidRegistrationObject();
// --------------------------------------------------------------------------- 

reg_f3d<PrecisionTYPE> *QmitkNiftyRegView::CreateNonRigidRegistrationObject( mitk::Image *mitkSourceImage, 
										mitk::Image *mitkTargetImage, 
										mitk::Image *mitkTargetMaskImage )
{
  // Get nifti versions of the images

  if ( m_ReferenceImage ) nifti_image_free( m_ReferenceImage );
  m_ReferenceImage = ConvertMitkImageToNifti( mitkTargetImage );

  if ( m_FloatingImage ) nifti_image_free( m_FloatingImage );
  m_FloatingImage = ConvertMitkImageToNifti( mitkSourceImage );

#if 0
  nifti_set_filenames( m_ReferenceImage,"f3dReference.nii",0,0 );
  nifti_image_write( m_ReferenceImage );

  nifti_set_filenames( m_FloatingImage,"f3dFloating.nii",0,0 );
  nifti_image_write( m_FloatingImage );
#endif

  // Check the dimensions of the images

  reg_checkAndCorrectDimension( m_ReferenceImage );
  reg_checkAndCorrectDimension( m_FloatingImage );

  // Set the reference mask image 

  if ( mitkTargetMaskImage )
  {

    if ( m_ReferenceMaskImage ) nifti_image_free( m_ReferenceMaskImage );
    m_ReferenceMaskImage = ConvertMitkImageToNifti( mitkTargetMaskImage );

    reg_checkAndCorrectDimension( m_ReferenceMaskImage );

    // check the dimensions

    for ( int i=1; i<=m_ReferenceImage->dim[0]; i++ )
    {
    
      if ( m_ReferenceImage->dim[i] != m_ReferenceMaskImage->dim[i] ) 
      {
	fprintf(stderr,"* ERROR The reference image and its mask do not have the same dimension\n");
	return 0;
      }
    }
  }

  // Read the input control point grid image

  if ( ! m_RegF3dParameters.inputControlPointGridName.isEmpty() ) 
  {

    if ( m_ControlPointGridImage ) nifti_image_free( m_ControlPointGridImage );
    m_ControlPointGridImage = nifti_image_read( m_RegF3dParameters.inputControlPointGridName
					      .toStdString().c_str(), true );

    if ( m_ControlPointGridImage == NULL ) 
    {
      fprintf(stderr, 
	      "Error when reading the input control point grid image %s\n",
	      m_RegF3dParameters.inputControlPointGridName.toStdString().c_str());
      return 0;
    }
    
    reg_checkAndCorrectDimension( m_ControlPointGridImage );
  }

  // Read the affine transformation

  mat44 *affineTransformation = NULL;

  if ( ( ! m_FlagDoInitialRigidReg ) &&
       m_FlagInputAffine && 
       ( ! m_InputAffineName.isEmpty() ) ) 
  {
    
    affineTransformation = (mat44 *) malloc( sizeof( mat44 ) );
    
    // Check first if the specified affine file exist
    
    if ( FILE *aff = fopen( m_InputAffineName.toStdString().c_str(), "r") ) 
    {
      fclose( aff );
    }
    else 
    {
      fprintf( stderr, "The specified input affine file (%s) can not be read\n",
	       m_InputAffineName.toStdString().c_str() );
      return 0;
    }
    
    reg_tool_ReadAffineFile( affineTransformation,
			     m_ReferenceImage,
			     m_FloatingImage,
			     strdup( m_InputAffineName.toStdString().c_str() ),
			     m_FlagFlirtAffine );
  }

  // Create the reg_f3d object

  reg_f3d<PrecisionTYPE> *REG = NULL;

#ifdef _USE_CUDA

  CUdevice dev;
  CUcontext ctx;

  if ( m_RegF3dParameters.useGPU )
  {

    if ( m_RegF3dParameters.linearEnergyWeight0 ||
	 m_RegF3dParameters.linearEnergyWeight1 ) {

      fprintf(stderr,"NiftyReg ERROR CUDA] The linear elasticity has not been implemented with CUDA yet. Exit.\n");
      exit(0);
    }

    if ( ( m_ReferenceImage->dim[4] == 1 && 
	   m_FloatingImage->dim[4]  == 1 ) || 
	 ( m_ReferenceImage->dim[4] == 2 &&
	   m_FloatingImage->dim[4]  == 2 ) ) {

      // The CUDA card is setup

      cuInit(0);

      struct cudaDeviceProp deviceProp;     
      int device_count = 0;      

      cudaGetDeviceCount( &device_count );

      int device = m_RegF3dParameters.cardNumber;
      
      if ( m_RegF3dParameters.cardNumber == -1 ) 
      {
	
	// following code is from cutGetMaxGflopsDeviceId()
	
	int max_gflops_device = 0;
	int max_gflops = 0;
	int current_device = 0;

	while ( current_device < device_count ) 
	{
	  cudaGetDeviceProperties( &deviceProp, current_device );

	  int gflops = deviceProp.multiProcessorCount * deviceProp.clockRate;

	  if ( gflops > max_gflops ) 
	  {
	    max_gflops = gflops;
	    max_gflops_device = current_device;
	  }
	  ++current_device;
	}
	device = max_gflops_device;
      }
      
      NR_CUDA_SAFE_CALL(cudaSetDevice( device ));
      NR_CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device ));

      cuDeviceGet(&dev,device);
      cuCtxCreate(&ctx, 0, dev);

      if ( deviceProp.major < 1 ) 
      {
	printf("[NiftyReg ERROR CUDA] The specified graphical card does not exist.\n");
	return 0;
      }
       
      REG = new reg_f3d_gpu<PrecisionTYPE>(m_ReferenceImage->nt, m_FloatingImage->nt);

    }
    else
    {
      fprintf(stderr,
	      "[NiftyReg ERROR] The GPU implementation only handles "
	      "1 to 1 or 2 to 2 image(s) registration\n");
      exit(1);
    }
  }
  
  else

#endif // _USE_CUDA
    
  {
    
    REG = new reg_f3d<PrecisionTYPE>( m_ReferenceImage->nt, 
				      m_FloatingImage->nt );

  }

  // Set the reg_f3d parameters

  REG->SetReferenceImage( m_ReferenceImage );
  REG->SetFloatingImage( m_FloatingImage );

  REG->PrintOutInformation();

  if ( mitkTargetMaskImage )
    REG->SetReferenceMask( m_ReferenceMaskImage );

  if ( m_ControlPointGridImage != NULL )
    REG->SetControlPointGridImage( m_ControlPointGridImage );

  if ( affineTransformation != NULL )
    REG->SetAffineTransformation( affineTransformation );
  
  REG->SetBendingEnergyWeight( m_RegF3dParameters.bendingEnergyWeight );
    
  REG->SetLinearEnergyWeights( m_RegF3dParameters.linearEnergyWeight0,
			       m_RegF3dParameters.linearEnergyWeight1 );
  
  REG->SetJacobianLogWeight( m_RegF3dParameters.jacobianLogWeight );
  
  if ( m_RegF3dParameters.jacobianLogApproximation )
    REG->ApproximateJacobianLog();
  else 
    REG->DoNotApproximateJacobianLog();

  REG->ApproximateParzenWindow();

  REG->SetMaximalIterationNumber( m_RegF3dParameters.maxiterationNumber );

  REG->SetReferenceSmoothingSigma( m_TargetSigmaValue );
  REG->SetFloatingSmoothingSigma( m_SourceSigmaValue );

  // NB: -std::numeric_limits<PrecisionTYPE>::max() is a special value which 
  // indicates the maximum value for ThresholdUp and the minimum for ThresholdLow.

  if ( m_RegF3dParameters.referenceThresholdUp == -std::numeric_limits<PrecisionTYPE>::max() )
    REG->SetReferenceThresholdUp( 0, std::numeric_limits<PrecisionTYPE>::max() );
  else
    REG->SetReferenceThresholdUp( 0, m_RegF3dParameters.referenceThresholdUp );

  REG->SetReferenceThresholdLow( 0, m_RegF3dParameters.referenceThresholdLow );

  if ( m_RegF3dParameters.floatingThresholdUp == -std::numeric_limits<PrecisionTYPE>::max() )
    REG->SetFloatingThresholdUp( 0, std::numeric_limits<PrecisionTYPE>::max() );
  else
    REG->SetFloatingThresholdUp( 0, m_RegF3dParameters.floatingThresholdUp );

  REG->SetFloatingThresholdLow( 0, m_RegF3dParameters.floatingThresholdLow );

  REG->SetReferenceBinNumber( 0, m_RegF3dParameters.referenceBinNumber );
  REG->SetFloatingBinNumber( 0, m_RegF3dParameters.floatingBinNumber );
  
  if ( m_RegF3dParameters.warpedPaddingValue == -std::numeric_limits<PrecisionTYPE>::max() )
    REG->SetWarpedPaddingValue( std::numeric_limits<PrecisionTYPE>::quiet_NaN() );
  else
    REG->SetWarpedPaddingValue( m_RegF3dParameters.warpedPaddingValue );

  for ( unsigned int s=0; s<3; s++ )
    REG->SetSpacing( s, m_RegF3dParameters.spacing[s] );

  REG->SetLevelNumber( m_LevelNumber );
  REG->SetLevelToPerform( m_Level2Perform );

  REG->SetGradientSmoothingSigma( m_RegF3dParameters.gradientSmoothingSigma );

  if ( m_RegF3dParameters.similarity == SSD_SIMILARITY )
    REG->UseSSD();
  else
    REG->DoNotUseSSD();

  if ( m_RegF3dParameters.similarity == KLDIV_SIMILARITY )
    REG->UseKLDivergence();
  else 
    REG->DoNotUseKLDivergence();

  if ( m_RegF3dParameters.useConjugate )
    REG->UseConjugateGradient();
  else 
    REG->DoNotUseConjugateGradient();

  if ( m_RegF3dParameters.noPyramid )
    REG->DoNotUsePyramidalApproach();

  if ( m_RegF3dParameters.interpolation == CUBIC_INTERPOLATION )
    REG->UseCubicSplineInterpolation();
  else if ( m_RegF3dParameters.interpolation == LINEAR_INTERPOLATION )
    REG->UseLinearInterpolation();
  else if ( m_RegF3dParameters.interpolation == NEAREST_INTERPOLATION )
    REG->UseNeareatNeighborInterpolation();


    // Run the registration
#ifdef _USE_CUDA
    if (m_RegF3dParameters.useGPU && m_RegF3dParameters.checkMem) {
        size_t free, total, requiredMemory = REG->CheckMemoryMB_f3d();
        cuMemGetInfo(&free, &total);
        printf("[NiftyReg CUDA] The required memory to run the registration is %lu Mb\n",
               (unsigned long int)requiredMemory);
        printf("[NiftyReg CUDA] The GPU card has %lu Mb from which %lu Mb are currenlty free\n",
               (unsigned long int)total/(1024*1024), (unsigned long int)free/(1024*1024));
    }
#endif

    return REG;
}


// ---------------------------------------------------------------------------
// OnNumberOfLevelsSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNumberOfLevelsSpinBoxValueChanged(int value)
{
  m_LevelNumber = value;

  if ( m_LevelNumber < m_Level2Perform )
  {
    m_Level2Perform = m_LevelNumber;
    m_Controls.m_LevelsToPerformSpinBox->setValue( m_Level2Perform );
  }

  Modified();
}


// ---------------------------------------------------------------------------
// OnLevelsToPerformSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnLevelsToPerformSpinBoxValueChanged(int value)
{
  m_Level2Perform = value;

  if ( m_Level2Perform > m_LevelNumber )
  {
    m_LevelNumber = m_Level2Perform;
    m_Controls.m_NumberOfLevelsSpinBox->setValue( m_LevelNumber );
  }

  Modified();
}


// ---------------------------------------------------------------------------
// OnSmoothSourceImageDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnSmoothSourceImageDoubleSpinBoxValueChanged(double value)
{
  m_SourceSigmaValue = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnSmoothTargetImageDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnSmoothTargetImageDoubleSpinBoxValueChanged(double value)
{
  m_TargetSigmaValue = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnNoSmoothingPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNoSmoothingPushButtonPressed( void )
{
  m_SourceSigmaValue = 0.;
  m_Controls.m_SmoothSourceImageDoubleSpinBox->setValue( m_SourceSigmaValue );

  m_TargetSigmaValue = 0.;
  m_Controls.m_SmoothTargetImageDoubleSpinBox->setValue( m_TargetSigmaValue );

  Modified();
}


// ---------------------------------------------------------------------------
// OnDoBlockMatchingOnlyRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnDoBlockMatchingOnlyRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_FlagDoInitialRigidReg = true;
    m_FlagDoNonRigidReg = false;
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
    m_FlagDoInitialRigidReg = false;
    m_FlagDoNonRigidReg = true;
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
    m_FlagDoInitialRigidReg = true;
    m_FlagDoNonRigidReg = true;
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnInputAffineCheckBoxStateChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnInputAffineCheckBoxToggled( bool checked )
{
  m_FlagInputAffine = checked;
  Modified();
}


// ---------------------------------------------------------------------------
// OnInputAffineBrowsePushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnInputAffineBrowsePushButtonPressed( void )
{
  m_InputAffineName = 
    QFileDialog::getOpenFileName(NULL, 
				 tr("Select affine transformation file"), 
				 QDir::currentPath(), 
				 tr("Affine transform file (*.txt *.tfm);;"
				    "Any file (*)"));

  if ( ! m_InputAffineName.isEmpty() )
  {
    m_FlagInputAffine = true;

    m_Controls.m_InputAffineFileNameLineEdit->setText( m_InputAffineName );

  }
  else
  {
    m_FlagInputAffine = false;

    m_Controls.m_InputAffineFileNameLineEdit->setText( QString( "" ) );
  }

  m_Controls.m_InitialAffineTransformationGroupBox->setChecked( m_FlagInputAffine );

  Modified();
}


// ---------------------------------------------------------------------------
// OnInputFlirtCheckBoxStateChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnInputFlirtCheckBoxStateChanged( int state )
{
  if ( state == Qt::Checked )
  {
    m_FlagFlirtAffine = true;
  }
  else
  {
    m_FlagFlirtAffine = false;
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
    m_RegAladinParameters.alignCenterFlag = false;
  }
  else
  {
    m_RegAladinParameters.alignCenterFlag = true;
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
    m_RegAladinParameters.regnType = RIGID_ONLY;
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
    m_RegAladinParameters.regnType = RIGID_THEN_AFFINE;
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
    m_RegAladinParameters.regnType = DIRECT_AFFINE;
    Modified();
  }
}


// ---------------------------------------------------------------------------
// OnAladinIterationsMaxSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnAladinIterationsMaxSpinBoxValueChanged(int value)
{
  m_RegAladinParameters.maxiterationNumber = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnPercentBlockSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnPercentBlockSpinBoxValueChanged(int value)
{
  m_RegAladinParameters.block_percent_to_use = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnPercentInliersSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnPercentInliersSpinBoxValueChanged(int value)
{
  m_RegAladinParameters.inlier_lts = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnAladinInterpolationNearestRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnAladinInterpolationNearestRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegAladinParameters.interpolation = NEAREST_INTERPOLATION;
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
    m_RegAladinParameters.interpolation = LINEAR_INTERPOLATION;
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
    m_RegAladinParameters.interpolation = CUBIC_INTERPOLATION;
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
    m_RegF3dParameters.inputControlPointGridFlag = true;
  }
  else
  {
    m_RegF3dParameters.inputControlPointGridFlag = false;
  }    
  Modified();
}


// ---------------------------------------------------------------------------
// OnNonRigidInputControlPointBrowsePushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNonRigidInputControlPointBrowsePushButtonPressed( void )
{
  m_RegF3dParameters.inputControlPointGridName = 
    QFileDialog::getOpenFileName(NULL, 
				 tr("Select spline control point nifti file"), 
				 QDir::currentPath(), 
				 tr("Control point file (*.nii);;"
				    "Any file (*)"));

  if ( ! m_RegF3dParameters.inputControlPointGridName.isEmpty() )
  {
    m_RegF3dParameters.inputControlPointGridFlag = true;

    m_Controls.m_NonRigidInputControlPointFileNameLineEdit->setText( m_RegF3dParameters.
								     inputControlPointGridName );
  }
  else
  {
    m_RegF3dParameters.inputControlPointGridFlag = false;

    m_Controls.m_NonRigidInputControlPointFileNameLineEdit->setText( m_RegF3dParameters.
								     inputControlPointGridName );
  }

  m_Controls.m_NonRigidInputControlPointCheckBox->setChecked( m_RegF3dParameters.
							      inputControlPointGridFlag );

  Modified();
}


// ---------------------------------------------------------------------------
// OnLowerThresholdTargetImageDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnLowerThresholdTargetImageDoubleSpinBoxValueChanged(double value)
{
 m_RegF3dParameters.referenceThresholdLow = value;

 if ( ( m_RegF3dParameters.referenceThresholdLow != -std::numeric_limits<PrecisionTYPE>::max() )
      && ( m_RegF3dParameters.referenceThresholdUp != -std::numeric_limits<PrecisionTYPE>::max() )
      && ( value > m_RegF3dParameters.referenceThresholdUp ) )
 {
   m_RegF3dParameters.referenceThresholdUp = -std::numeric_limits<PrecisionTYPE>::max();

   m_Controls.m_UpperThresholdTargetImageDoubleSpinBox
     ->setValue( m_RegF3dParameters.referenceThresholdUp );
 }

 Modified();
}


// ---------------------------------------------------------------------------
// OnLowerThresholdTargetImageAutoPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnLowerThresholdTargetImageAutoPushButtonPressed( void )
{
  m_RegF3dParameters.referenceThresholdLow = -std::numeric_limits<PrecisionTYPE>::max();

  m_Controls.m_LowerThresholdTargetImageDoubleSpinBox
    ->setValue( m_RegF3dParameters.referenceThresholdLow );

  Modified();
}


// ---------------------------------------------------------------------------
// OnUpperThresholdTargetImageDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnUpperThresholdTargetImageDoubleSpinBoxValueChanged(double value)
{
 m_RegF3dParameters.referenceThresholdUp = value;

 if ( ( m_RegF3dParameters.referenceThresholdUp != -std::numeric_limits<PrecisionTYPE>::max() ) 
      && ( m_RegF3dParameters.referenceThresholdLow != -std::numeric_limits<PrecisionTYPE>::max() ) 
      && ( value < m_RegF3dParameters.referenceThresholdLow ) )
 {
   m_RegF3dParameters.referenceThresholdLow = -std::numeric_limits<PrecisionTYPE>::max();

   m_Controls.m_LowerThresholdTargetImageDoubleSpinBox
     ->setValue( m_RegF3dParameters.referenceThresholdLow );
 }

 Modified();
}


// ---------------------------------------------------------------------------
// OnUpperThresholdTargetImageAutoPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnUpperThresholdTargetImageAutoPushButtonPressed( void )
{
  m_RegF3dParameters.referenceThresholdUp = -std::numeric_limits<PrecisionTYPE>::max();

  m_Controls.m_UpperThresholdTargetImageDoubleSpinBox
    ->setValue( m_RegF3dParameters.referenceThresholdUp );

  Modified();
}


// ---------------------------------------------------------------------------
// OnLowerThresholdSourceImageDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnLowerThresholdSourceImageDoubleSpinBoxValueChanged(double value)
{
  m_RegF3dParameters.floatingThresholdLow = value;

  if ( ( m_RegF3dParameters.floatingThresholdLow != -std::numeric_limits<PrecisionTYPE>::max() ) 
       && ( m_RegF3dParameters.floatingThresholdUp != -std::numeric_limits<PrecisionTYPE>::max() ) 
       && ( value > m_RegF3dParameters.floatingThresholdUp ) )
 {
   m_RegF3dParameters.floatingThresholdUp = -std::numeric_limits<PrecisionTYPE>::max();

   m_Controls.m_UpperThresholdSourceImageDoubleSpinBox
     ->setValue( m_RegF3dParameters.floatingThresholdUp );
 }

  Modified();
}


// ---------------------------------------------------------------------------
// OnLowerThresholdSourceImageAutoPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnLowerThresholdSourceImageAutoPushButtonPressed( void )
{
  m_RegF3dParameters.floatingThresholdLow = -std::numeric_limits<PrecisionTYPE>::max();

  m_Controls.m_LowerThresholdSourceImageDoubleSpinBox
    ->setValue( m_RegF3dParameters.floatingThresholdLow );

  Modified();
}


// ---------------------------------------------------------------------------
// OnUpperThresholdSourceImageDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnUpperThresholdSourceImageDoubleSpinBoxValueChanged(double value)
{
  m_RegF3dParameters.floatingThresholdUp = value;

  if ( ( m_RegF3dParameters.floatingThresholdUp != -std::numeric_limits<PrecisionTYPE>::max() ) 
       && ( m_RegF3dParameters.floatingThresholdLow != -std::numeric_limits<PrecisionTYPE>::max() ) 
       && ( value < m_RegF3dParameters.floatingThresholdLow ) )
 {
   m_RegF3dParameters.floatingThresholdLow = -std::numeric_limits<PrecisionTYPE>::max();

   m_Controls.m_LowerThresholdSourceImageDoubleSpinBox
     ->setValue( m_RegF3dParameters.floatingThresholdLow );
 }

  Modified();
}


// ---------------------------------------------------------------------------
// OnUpperThresholdSourceImageAutoPushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnUpperThresholdSourceImageAutoPushButtonPressed( void )
{
  m_RegF3dParameters.floatingThresholdUp = -std::numeric_limits<PrecisionTYPE>::max();

  m_Controls.m_UpperThresholdSourceImageDoubleSpinBox
    ->setValue( m_RegF3dParameters.floatingThresholdUp );

  Modified();
}


// ---------------------------------------------------------------------------
// OnControlPointSpacingXDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnControlPointSpacingXDoubleSpinBoxValueChanged(double value)
{
  m_RegF3dParameters.spacing[0] = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnControlPointSpacingYDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnControlPointSpacingYDoubleSpinBoxValueChanged(double value)
{
  m_RegF3dParameters.spacing[1] = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnControlPointSpacingZDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnControlPointSpacingZDoubleSpinBoxValueChanged(double value)
{
  m_RegF3dParameters.spacing[2] = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnNumberSourceHistogramBinsSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNumberSourceHistogramBinsSpinBoxValueChanged(int value)
{
  m_RegF3dParameters.floatingBinNumber = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnNumberTargetHistogramBinsSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNumberTargetHistogramBinsSpinBoxValueChanged(int value)
{
  m_RegF3dParameters.referenceBinNumber = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnWeightBendingEnergyDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnWeightBendingEnergyDoubleSpinBoxValueChanged(double value)
{
  m_RegF3dParameters.bendingEnergyWeight = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnWeightLogJacobianDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnWeightLogJacobianDoubleSpinBoxValueChanged(double value)
{
  m_RegF3dParameters.jacobianLogWeight = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnLinearEnergyWeightsDoubleSpinBox_1ValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnLinearEnergyWeightsDoubleSpinBox_1ValueChanged(double value)
{
  m_RegF3dParameters.linearEnergyWeight0 = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnLinearEnergyWeightsDoubleSpinBox_2ValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnLinearEnergyWeightsDoubleSpinBox_2ValueChanged(double value)
{
  m_RegF3dParameters.linearEnergyWeight1 = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnApproxJacobianLogCheckBoxStateChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnApproxJacobianLogCheckBoxStateChanged( int state )
{
  if ( state == Qt::Checked )
  {
    m_RegF3dParameters.jacobianLogApproximation = true;
  }
  else
  {
    m_RegF3dParameters.jacobianLogApproximation = false;
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
    m_RegF3dParameters.similarity = NMI_SIMILARITY;
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
    m_RegF3dParameters.similarity = SSD_SIMILARITY;
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
    m_RegF3dParameters.similarity = KLDIV_SIMILARITY;
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
    m_RegF3dParameters.useConjugate = false;
  }
  else
  {
    m_RegF3dParameters.useConjugate = true;
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
    m_RegF3dParameters.noPyramid = false;
  }
  else
  {
    m_RegF3dParameters.noPyramid = true;
  }    
  Modified();
}


// ---------------------------------------------------------------------------
// OnNonRigidIterationsMaxSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNonRigidIterationsMaxSpinBoxValueChanged(int value)
{
  m_RegF3dParameters.maxiterationNumber = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnSmoothingMetricDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnSmoothingMetricDoubleSpinBoxValueChanged(double value)
{
  m_RegF3dParameters.gradientSmoothingSigma = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnWarpedPaddingValueDoubleSpinBoxValueChanged();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnWarpedPaddingValueDoubleSpinBoxValueChanged(double value)
{
  m_RegF3dParameters.warpedPaddingValue = value;
  Modified();
}


// ---------------------------------------------------------------------------
// OnWarpedPaddingValuePushButtonPressed();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnWarpedPaddingValuePushButtonPressed( void )
{
  m_RegF3dParameters.warpedPaddingValue =  -std::numeric_limits<PrecisionTYPE>::max();
  m_Controls.m_WarpedPaddingValueDoubleSpinBox->setValue( m_RegF3dParameters.warpedPaddingValue );

  Modified();
}


// ---------------------------------------------------------------------------
// OnNonRigidNearestInterpolationRadioButtonToggled();
// --------------------------------------------------------------------------- 

void QmitkNiftyRegView::OnNonRigidNearestInterpolationRadioButtonToggled(bool checked)
{
  if ( checked )
  {
    m_RegF3dParameters.interpolation = NEAREST_INTERPOLATION;
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
    m_RegF3dParameters.interpolation = LINEAR_INTERPOLATION;
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
    m_RegF3dParameters.interpolation = CUBIC_INTERPOLATION;
    Modified();
  }
}


