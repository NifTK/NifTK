/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkNiftyRegView_h
#define QmitkNiftyRegView_h

//#undef _USE_CUDA

#include "ui_QmitkNiftyRegViewControls.h"
#include "berryISelectionListener.h"
#include "QmitkAbstractView.h"

// ITK
#include <itkMultiThreader.h>

#include "NiftyRegParameters.h"

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

    void OnAladinUseSymmetricAlgorithmCheckBoxStateChanged( int state );
    
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
    void OnSaveTransformationPushButtonPressed( void );
    void OnExecutePushButtonPressed( void );

    void OnSaveRegistrationParametersPushButtonPressed( void );
    void OnLoadRegistrationParametersPushButtonPressed( void );

    friend void UpdateProgressBar( float pcntProgress, void *param );

    friend ITK_THREAD_RETURN_TYPE ExecuteRegistration( void *param );


  protected:

    /// \brief Get the DataNode with a specific name. If not found return 0.
    mitk::DataNode::Pointer GetDataNode( QString searchName );

    /// \brief Get the list of data nodes from the data manager
    mitk::DataStorage::SetOfObjects::ConstPointer GetNodes();
    
    /// \brief Update the Aladin result/transformed image filename
    void UpdateAladinResultImageFilename();
    /// \brief Update the non-rigid result/transformed image filename
    void UpdateNonRigidResultImageFilename();

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

    /// \brief Save the registration parameters (as a shell-script command line)
    void WriteRegistrationParametersToFile( QString &filename );

    /// \brief Read the registration parameters (as a shell-script command line)
    void ReadRegistrationParametersFromFile( QString &filename );

    /// \brief Print the object
    void PrintSelf( std::ostream& os );

    /// \brief Function called whenever the object is modified
    void Modified();


    /// \brief The specific controls for this widget
    Ui::QmitkNiftyRegViewControls m_Controls;

    /// \brief Flag indicating whether any factors influencing the segmentation have been modified
    bool m_Modified;


    /** The current progress bar offset (0 < x < 100%) to enable progress to
     * be divided between multiple processes. */
    float m_ProgressBarOffset;
    /** The current progress bar range (0 < x < 100%) to enable progress to
     * be divided between multiple processes. */
    float m_ProgressBarRange;

    /// The registration parameters
    NiftyRegParameters<PrecisionTYPE> m_RegParameters;

    /// The 'reg_aladin' registration object
    reg_aladin<PrecisionTYPE> *m_RegAladin;

    /// The 'reg_f3d' registration object
    reg_f3d<PrecisionTYPE> *m_RegNonRigid;

};


void UpdateProgressBar( float pcntProgress, void *param );
ITK_THREAD_RETURN_TYPE ExecuteRegistration( void *param );

#endif // QmitkNiftyRegView_h

