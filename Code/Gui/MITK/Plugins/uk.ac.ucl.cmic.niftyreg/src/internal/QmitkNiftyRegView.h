/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef QmitkNiftyRegView_h
#define QmitkNiftyRegView_h

#undef _USE_CUDA

#include "ui_QmitkNiftyRegViewControls.h"
#include "berryISelectionListener.h"
#include "QmitkAbstractView.h"

// ITK
#include <itkMultiThreader.h>


#include "_reg_aladin.h"
#include "_reg_tools.h"

#include "_reg_f3d2.h"

#ifdef _USE_CUDA
#include "_reg_f3d_gpu.h"
#endif


#ifdef _USE_NR_DOUBLE
#define PrecisionTYPE double
#else
#define PrecisionTYPE float
#endif


/**
 * \class QmitkNiftyRegView
 * \brief GUI interface to enable the user to run the NiftyReg registration algorithm.
 * \ingroup uk.ac.ucl.cmic.niftyreg
*/


class QmitkNiftyRegView : public QmitkAbstractView
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT
  
  friend class RegistrationExecution;

  public:  

    static const std::string VIEW_ID;

    QmitkNiftyRegView();
    virtual ~QmitkNiftyRegView();

    void OnNodeAdded(const mitk::DataNode* node);
    void OnNodeRemoved(const mitk::DataNode* node);
    void OnNodeChanged(const mitk::DataNode* node);


  protected slots:

    /** \brief Slot for the source image combo box which when clicked 
	should list all the currently loaded images. */

    void OnSourceImageComboBoxChanged( int index );
    void OnTargetImageComboBoxChanged( int index );

    void OnTargetMaskImageComboBoxChanged( int index );

    // Multi-Scale Options
    
    void OnNumberOfLevelsSpinBoxValueChanged( int value );
    void OnLevelsToPerformSpinBoxValueChanged( int value );
    
    // Input Image Options
    
    void OnSmoothSourceImageDoubleSpinBoxValueChanged( double value );
    void OnSmoothTargetImageDoubleSpinBoxValueChanged( double value );
    void OnNoSmoothingPushButtonPressed( void );

    void OnDoBlockMatchingOnlyRadioButtonToggled( bool checked );
    void OnDoNonRigidOnlyRadioButtonToggled( bool checked );
    void OnDoBlockMatchingThenNonRigidRadioButtonToggled( bool checked );

    // Initial Affine Transformation

    void OnInputAffineCheckBoxToggled( bool checked );
    void OnInputAffineBrowsePushButtonPressed( void );

    void OnInputFlirtCheckBoxStateChanged( int state );

     // Aladin - Initialisation

   void OnUseNiftyHeaderCheckBoxStateChanged( int state );

    // Aladin - Method

    void OnRigidOnlyRadioButtonToggled( bool checked );
    void OnRigidThenAffineRadioButtonToggled( bool checked );
    void OnDirectAffineRadioButtonToggled( bool checked );
    
    void OnAladinIterationsMaxSpinBoxValueChanged( int value );
    void OnPercentBlockSpinBoxValueChanged( int value );
    void OnPercentInliersSpinBoxValueChanged( int value );

    // Aladin - Advanced

    void OnAladinInterpolationNearestRadioButtonToggled( bool checked );
    void OnAladinInterpolationLinearRadioButtonToggled( bool checked );
    void OnAladinInterpolationCubicRadioButtonToggled( bool checked );

    
    
    // Non-Rigid - Initialisation
    
    void OnNonRigidInputControlPointCheckBoxStateChanged( int state );
    void OnNonRigidInputControlPointBrowsePushButtonPressed( void );


    // Non-Rigid - Input Image
    
    void OnLowerThresholdTargetImageDoubleSpinBoxValueChanged( double value );
    void OnUpperThresholdTargetImageDoubleSpinBoxValueChanged( double value );

    void OnUpperThresholdTargetImageAutoPushButtonPressed( void );
    void OnLowerThresholdTargetImageAutoPushButtonPressed( void );

    void OnLowerThresholdSourceImageDoubleSpinBoxValueChanged( double value );
    void OnUpperThresholdSourceImageDoubleSpinBoxValueChanged( double value );

    void OnUpperThresholdSourceImageAutoPushButtonPressed( void );
    void OnLowerThresholdSourceImageAutoPushButtonPressed( void );

    // Non-Rigid - Spline
    
    void OnControlPointSpacingXDoubleSpinBoxValueChanged( double value );
    void OnControlPointSpacingYDoubleSpinBoxValueChanged( double value );
    void OnControlPointSpacingZDoubleSpinBoxValueChanged( double value );

    // Non-Rigid - Objective Function
    
    void OnNumberSourceHistogramBinsSpinBoxValueChanged( int value );
    void OnNumberTargetHistogramBinsSpinBoxValueChanged( int value );

    void OnWeightBendingEnergyDoubleSpinBoxValueChanged( double value );
    void OnWeightLogJacobianDoubleSpinBoxValueChanged( double value );

    void OnLinearEnergyWeightsDoubleSpinBox_1ValueChanged( double value );
    void OnLinearEnergyWeightsDoubleSpinBox_2ValueChanged( double value );

    void OnApproxJacobianLogCheckBoxStateChanged( int state );

    void OnSimilarityNMIRadioButtonToggled( bool checked );
    void OnSimilaritySSDRadioButtonToggled( bool checked );
    void OnSimilarityKLDivRadioButtonToggled( bool checked );
    
    // Non-Rigid - Optimisation

    void OnUseSimpleGradientAscentCheckBoxStateChanged( int state );
    void OnNonRigidIterationsMaxSpinBoxValueChanged( int value );
    void OnUsePyramidalCheckBoxStateChanged( int state );
    
    // Non-Rigid - Advanced

    void OnSmoothingMetricDoubleSpinBoxValueChanged( double value );
    void OnWarpedPaddingValueDoubleSpinBoxValueChanged( double value );
    void OnWarpedPaddingValuePushButtonPressed( void );

    void OnNonRigidNearestInterpolationRadioButtonToggled( bool checked );
    void OnNonRigidLinearInterpolationRadioButtonToggled( bool checked );
    void OnNonRigidCubicInterpolationRadioButtonToggled( bool checked );


    // Execution

    void OnCancelPushButtonPressed( void );
    void OnResetParametersPushButtonPressed( void );
    void OnSaveAsPushButtonPressed( void );
    void OnExecutePushButtonPressed( void );

    friend void UpdateProgressBar( float pcntProgress, void *param );

    friend ITK_THREAD_RETURN_TYPE ExecuteRegistration( void *param );


  protected:

    /// Deallocate the nifti images used in the registration
    void DeallocateImages( void );

    /// \brief Get the list of data nodes from the data manager
    mitk::DataStorage::SetOfObjects::ConstPointer GetNodes();

    /// \brief Called by framework, this method creates all the controls for this view
    virtual void CreateQtPartControl(QWidget *parent);

    /// \brief Set the default parameters
    virtual void SetDefaultParameters();

    /// \brief Set the default state of the GUI
    virtual void SetGuiToParameterValues();

    /// \brief Creation of the connections of widgets in this class and the slots in this class.
    virtual void CreateConnections();

    /// \brief Called by framework, sets the focus on a specific widget.
    virtual void SetFocus();

    /// \brief Create the Aladin registration object
    reg_aladin<PrecisionTYPE> *CreateAladinRegistrationObject( mitk::Image *mitkSourceImage, 
							       mitk::Image *mitkTargetImage, 
							       mitk::Image *mitkTargetMaskImage );

    /// \brief Create the Aladin registration object
    reg_f3d<PrecisionTYPE> *CreateNonRigidRegistrationObject( mitk::Image *mitkSourceImage, 
							      mitk::Image *mitkTargetImage, 
							      mitk::Image *mitkTargetMaskImage );
    

    /// \brief Print the object
    void PrintSelf( std::ostream& os );

    /// \brief Function called whenever the object is modified
    void Modified();

    /// \brief The specific controls for this widget
    Ui::QmitkNiftyRegViewControls m_Controls;

    /// \brief Flag indicating whether any factors influencing the segmentation have been modified
    bool m_Modified;


    /// \brief The number of multi-resolution levels
    int m_LevelNumber;
    /// \brief The number of (coarse to fine) multi-resolution levels to use 
    int m_Level2Perform;    

    // Smooth the target image using the specified sigma (mm) 
    float m_TargetSigmaValue;
    // Smooth the source image using the specified sigma (mm)
    float m_SourceSigmaValue;

    /// Flag indicating whether to do an initial rigid registration
    bool m_FlagDoInitialRigidReg;
    /// Flag indicating whether to do the non-rigid registration
    bool m_FlagDoNonRigidReg;

    /// The filename of the initial affine transformation
    QString m_InputAffineName;  // -inaff
    /// Flag indicating whether an initial affine transformation is specified
    bool m_FlagInputAffine;
    /// Flag indicating whether the initial affine transformation is FLIRT
    bool m_FlagFlirtAffine;   // -affFlirt


    /** The current progress bar offset (0 < x < 100%) to enable progress to
     * be divided between multiple processes. */
    float m_ProgressBarOffset;
    /** The current progress bar range (0 < x < 100%) to enable progress to
     * be divided between multiple processes. */
    float m_ProgressBarRange;

    /// Codes for interpolation type
    typedef enum {
      UNSET_INTERPOLATION = 0,
      NEAREST_INTERPOLATION = 1,
      LINEAR_INTERPOLATION = 2,
      CUBIC_INTERPOLATION = 3
    } InterpolationType;

    /// Codes for similarity measure type
    typedef enum {
      UNSET_SIMILARITY = 0,
      NMI_SIMILARITY = 1,
      SSD_SIMILARITY = 2,
      KLDIV_SIMILARITY = 3
    } SimilarityType;

    /// Codes for affine registration type
    typedef enum {
      UNSET_TRANSFORMATION = 0,
      RIGID_ONLY = 1,
      RIGID_THEN_AFFINE = 2,
      DIRECT_AFFINE = 3
    } AffineRegistrationType;

    /// The reference/target image
    nifti_image *m_ReferenceImage;
    /// The floating/source image
    nifti_image *m_FloatingImage;
    /// The reference/target mask image
    nifti_image *m_ReferenceMaskImage;
    /// The input control grid image
    nifti_image *m_ControlPointGridImage;

    // ---------------------------------------------------------------------------
    // Rigid/Affine Aladin Parameters
    // ---------------------------------------------------------------------------

    typedef struct {

      bool outputResultFlag;
      QString outputResultName; // -res

      bool outputAffineFlag;
      QString outputAffineName; // -aff

      // Aladin - Initialisation

      bool alignCenterFlag;        // -nac

      // Aladin - Method

      AffineRegistrationType regnType; // -rigOnly, -affDirect

      int maxiterationNumber;		// -maxit

      int block_percent_to_use; // -%v
      int inlier_lts;		// -%i

      // Aladin - Advanced

      InterpolationType interpolation;

    } RegAladinParametersType;

    /// \brief The 'reg_aladin' parameters
    RegAladinParametersType m_RegAladinParameters;


    // ---------------------------------------------------------------------------
    // Non-rigid Parameters
    // ---------------------------------------------------------------------------

    typedef struct {

      // Initial transformation options:
 
      bool inputControlPointGridFlag;
      QString inputControlPointGridName;// -incpp

      // Output options:
 
      QString outputControlPointGridName; // -cpp
      QString outputWarpedName;		// -res

      // Input image options:

      PrecisionTYPE referenceThresholdUp;  // -rLwTh
      PrecisionTYPE referenceThresholdLow; // -rUpTh 

      PrecisionTYPE floatingThresholdUp;   // -fLwTh
      PrecisionTYPE floatingThresholdLow;  // -fUpTh

      // Spline options:
 
      PrecisionTYPE spacing[3];   // -sx, -sy, -sz

      // Objective function options:
 
      unsigned int referenceBinNumber;   // -rbn
      unsigned int floatingBinNumber;    // -fbn

      PrecisionTYPE bendingEnergyWeight;   // -be

      PrecisionTYPE linearEnergyWeight0;   // -le 
      PrecisionTYPE linearEnergyWeight1;   // -le 

      PrecisionTYPE jacobianLogWeight;     // -jl 

      bool jacobianLogApproximation;       // -noAppJL

      SimilarityType similarity;           // -ssd, -kld 

      // Optimisation options:
 
      bool useConjugate;                   // -noConj
      int maxiterationNumber;              // -maxit
      bool noPyramid;                      // -nopy

      // GPU-related options:

      bool checkMem;   // -mem
      bool useGPU;     // -gpu
      int cardNumber;  // -card

      // Other options:

      InterpolationType interpolation;

      PrecisionTYPE gradientSmoothingSigma;  // -smoothGrad
      PrecisionTYPE warpedPaddingValue;      // -pad
      bool verbose;                          // -voff

    } RegF3dParametersType;
    
    /// \brief The 'reg_f3d' parameters
    RegF3dParametersType m_RegF3dParameters;


};


void UpdateProgressBar( float pcntProgress, void *param );
ITK_THREAD_RETURN_TYPE ExecuteRegistration( void *param );

#endif // QmitkNiftyRegView_h

