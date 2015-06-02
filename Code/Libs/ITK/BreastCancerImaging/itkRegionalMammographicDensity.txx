/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#include <itkRegionalMammographicDensity.h>

<<<<<<< HEAD
=======
#include <itkImageFileWriter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkSpatialObjectToImageFilter.h>
#include <itkGDCMImageIO.h>
#include <itkNeighborhoodIterator.h>
#include <itkCastImageFilter.h>
#include <itkLabelToRGBImageFilter.h>
#include <itkRelabelComponentImageFilter.h>
#include <itkScalarConnectedComponentImageFilter.h>
#include <itkSignedMaurerDistanceMapImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkInvertIntensityBetweenMaxAndMinImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkIntensityWindowingImageFilter.h>
#include <itkFlipImageFilter.h>
#include <itkScaleTransform.h>

namespace fs = boost::filesystem;

>>>>>>> Issue #4270: Modification of RegionalMammographicDensity to use MammogramRegistrationFilter

namespace itk
{


// --------------------------------------------------------------------------
// Constructor
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
RegionalMammographicDensity< InputPixelType, InputDimension >
::RegionalMammographicDensity()
{
<<<<<<< HEAD
=======
  m_ThresholdDiagnostic = 0;
  m_ThresholdPreDiagnostic = 0;
  m_ThresholdControl = 0;

  m_TumourLeft = 0;
  m_TumourRight = 0;
  m_TumourTop = 0;
  m_TumourBottom = 0;
 
  m_TumourDiameter = 0;
 
  m_DiagTumourRegionValue = 0;

  m_DiagTumourCenterIndex[0] = 0;
  m_DiagTumourCenterIndex[1] = 0;

  m_PreDiagTumourRegionValue = 0;
  m_ControlTumourRegionValue = 0;

  m_PreDiagCenterIndex[0] = 0;
  m_PreDiagCenterIndex[1] = 0;

  m_ControlCenterIndex[0] = 0;
  m_ControlCenterIndex[1] = 0;

  m_RegionSizeInMM = 10.;

  m_ImDiagnostic = 0;
  m_ImPreDiagnostic = 0;
  m_ImControl = 0;

  m_FlgOverwrite = false;
  m_FlgRegister = false;

  m_TypeOfInputImagesToRegister = RegistrationFilterType::REGISTER_DISTANCE_TRANSFORMS;


  m_FlgVerbose = false;
  m_FlgDebug = false;

  m_FlgRegisterNonRigid = false;


  m_RegistrationPreDiag = 0;
  m_RegistrationControl = 0;


  m_BreastSideDiagnostic    = LeftOrRightSideCalculatorType::UNKNOWN_BREAST_SIDE;
  m_BreastSidePreDiagnostic = LeftOrRightSideCalculatorType::UNKNOWN_BREAST_SIDE;
  m_BreastSideControl       = LeftOrRightSideCalculatorType::UNKNOWN_BREAST_SIDE;

  m_Gen.seed(static_cast<unsigned int>(std::time(0)));

  UnloadImages();
>>>>>>> Issue #4270: Modification of RegionalMammographicDensity to use MammogramRegistrationFilter
};


// --------------------------------------------------------------------------
// Destructor
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
RegionalMammographicDensity< InputPixelType, InputDimension >
::~RegionalMammographicDensity()
{
}


// --------------------------------------------------------------------------
// RunRegistration()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::RunRegistration( void )
{
  typename ImageType::Pointer imAffineRegistered;
  typename ImageType::Pointer imNonRigidRegistered;

  // The pre-diagnostic registration

<<<<<<< HEAD
  if ( this->m_ImDiagnostic && this->m_ImPreDiagnostic )
=======
  std::cout << std::endl
            << "    Input image directory: " << m_DirInput << std::endl
            << "    Output directory: "      << m_DirOutput << std::endl;

  if ( m_FlgVerbose )
    std::cout << "   Verbose output: YES" << std::endl;
  else
    std::cout << "   Verbose output: NO" << std::endl;

  if ( m_FlgOverwrite )
    std::cout << "   Overwrite output: YES" << std::endl;
  else
    std::cout << "   Overwrite output: NO" << std::endl;

  if ( m_FlgRegister )
    std::cout << "   Affine register the images: YES" << std::endl;
  else
    std::cout << "   Affine register the images: NO" << std::endl;

  if ( m_FlgRegisterNonRigid )
    std::cout << "   Non-rigidly register the images: YES" << std::endl;
  else
    std::cout << "   Non-rigidly register the images: NO" << std::endl;

  if ( m_FlgDebug )
    std::cout << "   Debug output: YES" << std::endl;
  else
    std::cout << "   Debug output: NO" << std::endl;


  if ( m_FlgVerbose )
>>>>>>> Issue #4270: Modification of RegionalMammographicDensity to use MammogramRegistrationFilter
  {
    this->WriteRegistrationDifferenceImage( this->m_FilePreDiagnostic,
                                      std::string( "_PreDiag2DiagDifference.jpg" ),
                                      "un-registered pre-diagnostic difference image",
                                      this->m_ImPreDiagnostic,
                                      this->m_DiagDictionary );

    this->m_RegistrationPreDiag =
      this->RegisterTheImages( this->m_ImPreDiagnostic,
                         this->m_FilePreDiagnosticRegn,
                         this->m_ImPreDiagnosticMask,

                         this->BuildOutputFilename( this->m_FileDiagnostic,
                                              "_PreDiagReg2Diag_AffineTransform.txt" ),
                         this->BuildOutputFilename( this->m_FileDiagnostic,
                                              "_PreDiagReg2Diag_AffineRegistered.nii.gz" ),

                         this->BuildOutputFilename( this->m_FileDiagnostic,
                                              "_PreDiagReg2Diag_NonRigidTransform.nii.gz" ),
                         this->BuildOutputFilename( this->m_FileDiagnostic,
                                              "_PreDiagReg2Diag_NonRigidRegistered.nii.gz" ) );

    imAffineRegistered   = this->m_RegistrationPreDiag->GetOutput( 0 );

    this->WriteRegistrationDifferenceImage( this->m_FilePreDiagnostic,
                                      std::string( "_PreDiagReg2DiagAffineDifference.jpg" ),
                                      "affine registered pre-diagnostic difference image",
                                      imAffineRegistered,
                                      this->m_DiagDictionary );

    imNonRigidRegistered = this->m_RegistrationPreDiag->GetOutput( 1 );

    this->WriteRegistrationDifferenceImage( this->m_FilePreDiagnostic,
                                      std::string( "_PreDiagReg2DiagNonRigidDifference.jpg" ),
                                      "non-rigidly registered pre-diagnostic difference image",
                                      imNonRigidRegistered,
                                      this->m_DiagDictionary );
  }

  // The control image registration

  if ( this->m_ImDiagnostic && this->m_ImControl )
  {
    this->WriteRegistrationDifferenceImage( this->m_FileControl,
                                      std::string( "_Control2DiagDifference.jpg" ),
                                      "un-registered control difference image",
                                      this->m_ImControl,
                                      this->m_DiagDictionary );

    this->m_RegistrationControl =
      this->RegisterTheImages( this->m_ImControl,
                         this->m_FileControlRegn,
                         this->m_ImControlMask,

                         this->BuildOutputFilename( this->m_FileDiagnostic,
                                              "_ControlReg2Diag_AffineTransform.txt" ),
                         this->BuildOutputFilename( this->m_FileDiagnostic,
                                              "_ControlReg2Diag_AffineRegistered.nii.gz" ),

                         this->BuildOutputFilename( this->m_FileDiagnostic,
                                              "_ControlReg2Diag_NonRigidTransform.nii.gz" ),
                         this->BuildOutputFilename( this->m_FileDiagnostic,
                                              "_ControlReg2Diag_NonRigidRegistered.nii.gz" ) );

    imAffineRegistered   = this->m_RegistrationControl->GetOutput( 0 );

    this->WriteRegistrationDifferenceImage( this->m_FileControl,
                                      std::string( "_ControlReg2DiagAffineDifference.jpg" ),
                                      "affine registered control difference image",
                                      imAffineRegistered,
                                      this->m_DiagDictionary );

    imNonRigidRegistered = this->m_RegistrationControl->GetOutput( 1 );

    this->WriteRegistrationDifferenceImage( this->m_FileControl,
                                      std::string( "_ControlReg2DiagNonRigidDifference.jpg" ),
                                      "non-rigidly  registered control difference image",
                                      imNonRigidRegistered,
                                      this->m_DiagDictionary );
  }

};


// --------------------------------------------------------------------------
// Compute()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::Compute()
{
  std::string fileMask;
  std::string fileRegnMask;

  std::string diagMaskSuffix(       "_DiagMask.nii.gz" );
  std::string preDiagMaskSuffix( "_PreDiagMask.nii.gz" );
  std::string controlMaskSuffix( "_ControlMask.nii.gz" );

  std::string diagRegnMaskSuffix(       "_DiagRegnMask.nii.gz" );
  std::string preDiagRegnMaskSuffix( "_PreDiagRegnMask.nii.gz" );
  std::string controlRegnMaskSuffix( "_ControlRegnMask.nii.gz" );


  // If this is a right mammogram then flip the tumour index in 'x'

  if ( this->m_BreastSideDiagnostic == LeftOrRightSideCalculatorType::RIGHT_BREAST_SIDE )
  {
    typename ImageType::SizeType
      imSizeInPixels = this->m_ImDiagnostic->GetLargestPossibleRegion().GetSize();

    if ( this->m_FlgVerbose )
    {
      std::cout << "This is a right mammogram so flipping the tumour location in 'x'"
                << std::endl;
    }

    this->m_TumourLeft = imSizeInPixels[0] - this->m_TumourLeft;
    this->m_TumourRight = imSizeInPixels[0] - this->m_TumourRight;
  }


  // The center of the diagnostic tumour

  this->m_DiagTumourCenterIndex[0] = (this->m_TumourLeft +  this->m_TumourRight) / 2;
  this->m_DiagTumourCenterIndex[1] = ( this->m_TumourTop + this->m_TumourBottom) / 2;


  // Generate the Diagnostic Mask

  if ( this->m_ImDiagnostic )
  {

    fileMask = this->BuildOutputFilename( this->m_FileDiagnostic, diagMaskSuffix );

    if ( niftk::FileExists( fileMask ) && ( ! this->m_FlgOverwrite ) )
    {
      typename ReaderType::Pointer reader = ReaderType::New();
      reader->SetFileName( fileMask );

      try
      {
        std::cout << "Reading the diagnostic mask: " << fileMask << std::endl;
        reader->Update();
      }

      catch (ExceptionObject &ex)
      {
        std::cerr << "ERROR: Could not read file: "
                  << fileMask << std::endl << ex << std::endl;
        throw( ex );
      }

      this->m_ImDiagnosticMask = reader->GetOutput();
      this->m_ImDiagnosticMask->DisconnectPipeline();

      // And registration version?

<<<<<<< HEAD
      if ( this->m_FlgRegister || this->m_FlgRegisterNonRigid )
      {
        this->m_FileDiagnosticRegnMask = this->BuildOutputFilename( this->m_FileDiagnostic, diagRegnMaskSuffix );
=======
template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::ReadImage( MammogramType mammoType ) 
{
  std::string fileInput;
  std::string fileOutput;
  std::string fileImage;
>>>>>>> Issue #4270: Modification of RegionalMammographicDensity to use MammogramRegistrationFilter

        if ( niftk::FileExists( this->m_FileDiagnosticRegnMask ) )
        {
          reader->SetFileName( this->m_FileDiagnosticRegnMask );

          try
          {
            std::cout << "Reading the diagnostic registration mask: " << this->m_FileDiagnosticRegnMask << std::endl;
            reader->Update();
          }

          catch (ExceptionObject &ex)
          {
            std::cerr << "ERROR: Could not read file: "
                      << this->m_FileDiagnosticRegnMask << std::endl << ex << std::endl;
            throw( ex );
          }

          this->m_ImDiagnosticRegnMask = reader->GetOutput();
          this->m_ImDiagnosticRegnMask->DisconnectPipeline();
        }
        else
        {
          itkExceptionMacro( << "ERROR: Cannot read diagnostic registration mask: " << this->m_FileDiagnosticRegnMask );
        }
      }
    }
    else
    {
      this->m_ImDiagnosticMask = this->MaskWithPolygon( Superclass::DIAGNOSTIC_MAMMO );

      this->template CastImageAndWriteToFile< unsigned char >( this->m_FileDiagnostic,
                                                diagMaskSuffix,
                                                "diagnostic mask",
                                                this->m_ImDiagnosticMask,
                                                this->m_DiagDictionary );
    }
  }

  // Generate the Pre-diagnostic Mask

  if ( this->m_ImPreDiagnostic )
  {

    fileMask = this->BuildOutputFilename( this->m_FilePreDiagnostic, preDiagMaskSuffix );

    if ( niftk::FileExists( fileMask ) && ( ! this->m_FlgOverwrite ) )
    {
      typename ReaderType::Pointer reader = ReaderType::New();
      reader->SetFileName( fileMask );

      try
      {
        std::cout << "Reading the pre-diagnostic mask: " << fileMask << std::endl;
        reader->Update();
      }

      catch (ExceptionObject &ex)
      {
        std::cerr << "ERROR: Could not read file: "
                  << fileMask << std::endl << ex << std::endl;
        throw( ex );
      }

      this->m_ImPreDiagnosticMask = reader->GetOutput();
      this->m_ImPreDiagnosticMask->DisconnectPipeline();

      // And registration version?

      if ( this->m_FlgRegister || this->m_FlgRegisterNonRigid )
      {
        fileRegnMask = this->BuildOutputFilename( this->m_FilePreDiagnostic, preDiagRegnMaskSuffix );

        if ( niftk::FileExists( fileRegnMask ) )
        {
          reader->SetFileName( fileRegnMask );

          try
          {
            std::cout << "Reading the pre-diagnostic registration mask: " << fileRegnMask << std::endl;
            reader->Update();
          }

          catch (ExceptionObject &ex)
          {
            std::cerr << "ERROR: Could not read file: "
                      << fileRegnMask << std::endl << ex << std::endl;
            throw( ex );
          }

          this->m_ImPreDiagnosticRegnMask = reader->GetOutput();
          this->m_ImPreDiagnosticRegnMask->DisconnectPipeline();
        }
        else
        {
          itkExceptionMacro( << "ERROR: Cannot read pre-diagnostic registration mask: " << fileRegnMask );
        }
      }
    }
    else
    {
      this->m_ImPreDiagnosticMask = this->MaskWithPolygon( Superclass::PREDIAGNOSTIC_MAMMO );

      this->template CastImageAndWriteToFile< unsigned char >( this->m_FilePreDiagnostic,
                                                preDiagMaskSuffix,
                                                "pre-diagnostic mask",
                                                this->m_ImPreDiagnosticMask,
                                                this->m_PreDiagDictionary );
    }
  }

<<<<<<< HEAD
  // Generate the Control Mask

  if ( this->m_ImControl )
=======
  if ( m_FlgDebug )
    std::cout << "  breast side: " 
              << LeftOrRightSideCalculatorType::BreastSideDescription[ leftOrRightCalculator->GetBreastSide() ] << std::endl;

  // Set values for each mammogram

  typename ImageType::SpacingType imSpacing;
  typename ImageType::SpacingType dcmSpacing;


  if ( mammoType == DIAGNOSTIC_MAMMO )
>>>>>>> Issue #4270: Modification of RegionalMammographicDensity to use MammogramRegistrationFilter
  {

<<<<<<< HEAD
    fileMask = this->BuildOutputFilename( this->m_FileControl, controlMaskSuffix );
=======
    imSpacing = m_ImDiagnostic->GetSpacing();
    dcmSpacing = GetImageResolutionFromDictionary( m_DiagDictionary );

    if ( ( ( imSpacing[0] != dcmSpacing[0] ) ||
           ( imSpacing[1] != dcmSpacing[1] ) ) &&
         ( ( imSpacing[0] == 1 ) ||
           ( imSpacing[1] != 1 ) ) )
    {
      m_ImDiagnostic->SetSpacing( dcmSpacing );

      fileOutput = 
        niftk::ConcatenatePath( m_DirOutput, 
                                niftk::ModifyImageFileSuffix( fileImage,
                                                              std::string( ".nii.gz" ) ) );

      itk::WriteImageToFile< ImageType >( fileOutput.c_str(), 
                                               "diagnostic image",  m_ImDiagnostic );      
      m_FileDiagnosticRegn = fileOutput;
    }
    else
    {
      m_FileDiagnosticRegn = fileInput;
    }
>>>>>>> Issue #4270: Modification of RegionalMammographicDensity to use MammogramRegistrationFilter

    if ( niftk::FileExists( fileMask ) && ( ! this->m_FlgOverwrite ) )
    {
      typename ReaderType::Pointer reader = ReaderType::New();
      reader->SetFileName( fileMask );

<<<<<<< HEAD
      try
      {
        std::cout << "Reading the control mask: " << fileMask << std::endl;
        reader->Update();
      }
=======
    imSpacing = m_ImPreDiagnostic->GetSpacing();
    dcmSpacing = GetImageResolutionFromDictionary( m_PreDiagDictionary );

    if ( ( ( imSpacing[0] != dcmSpacing[0] ) ||
           ( imSpacing[1] != dcmSpacing[1] ) ) &&
         ( ( imSpacing[0] == 1 ) ||
           ( imSpacing[1] != 1 ) ) )
    {
      m_ImPreDiagnostic->SetSpacing( dcmSpacing );

      fileOutput = 
        niftk::ConcatenatePath( m_DirOutput, 
                                niftk::ModifyImageFileSuffix( fileImage,
                                                              std::string( ".nii.gz" ) ) );

      itk::WriteImageToFile< ImageType >( fileOutput.c_str(), 
                                               "pre-diagnostic image",  m_ImPreDiagnostic );      
      m_FilePreDiagnosticRegn = fileOutput;
    }
    else
    {
      m_FilePreDiagnosticRegn = fileInput;
    }
>>>>>>> Issue #4270: Modification of RegionalMammographicDensity to use MammogramRegistrationFilter

      catch (ExceptionObject &ex)
      {
        std::cerr << "ERROR: Could not read file: "
                  << fileMask << std::endl << ex << std::endl;
        throw( ex );
      }

<<<<<<< HEAD
      this->m_ImControlMask = reader->GetOutput();
      this->m_ImControlMask->DisconnectPipeline();
=======
    imSpacing = m_ImControl->GetSpacing();
    dcmSpacing = GetImageResolutionFromDictionary( m_ControlDictionary );

    if ( ( ( imSpacing[0] != dcmSpacing[0] ) ||
           ( imSpacing[1] != dcmSpacing[1] ) ) &&
         ( ( imSpacing[0] == 1 ) ||
           ( imSpacing[1] != 1 ) ) )
    {
      m_ImControl->SetSpacing( dcmSpacing );

      fileOutput = 
        niftk::ConcatenatePath( m_DirOutput, 
                                niftk::ModifyImageFileSuffix( fileImage,
                                                              std::string( ".nii.gz" ) ) );

      itk::WriteImageToFile< ImageType >( fileOutput.c_str(), 
                                               "control image",  m_ImControl );      
      m_FileControlRegn = fileOutput;
    }
    else
    {
      m_FileControlRegn = fileInput;
    }
>>>>>>> Issue #4270: Modification of RegionalMammographicDensity to use MammogramRegistrationFilter

      // And registration version?

      if ( this->m_FlgRegister || this->m_FlgRegisterNonRigid )
      {
        fileRegnMask = this->BuildOutputFilename( this->m_FileControl, controlRegnMaskSuffix );

        if ( niftk::FileExists( fileRegnMask ) )
        {
          reader->SetFileName( fileRegnMask );

          try
          {
            std::cout << "Reading the control registration mask: " << fileRegnMask << std::endl;
            reader->Update();
          }

          catch (ExceptionObject &ex)
          {
            std::cerr << "ERROR: Could not read file: "
                      << fileRegnMask << std::endl << ex << std::endl;
            throw( ex );
          }

          this->m_ImControlRegnMask = reader->GetOutput();
          this->m_ImControlRegnMask->DisconnectPipeline();
        }
        else
        {
          itkExceptionMacro( << "ERROR: Cannot read control registration mask: " << fileRegnMask );
        }
      }
    }
    else
    {
      this->m_ImControlMask = this->MaskWithPolygon( Superclass::CONTROL_MAMMO );

      this->template CastImageAndWriteToFile< unsigned char >( this->m_FileControl,
							       controlMaskSuffix,
							       "control mask",
							       this->m_ImControlMask,
							       this->m_ControlDictionary );
    }
  }

  // Register the images?

  if ( this->m_FlgRegister )
  {
    RunRegistration();
  }


  // Calculate the diagnostic labels

  if ( this->m_ImDiagnostic )
  {
    if ( this->m_FlgVerbose )
      std::cout << "Computing diagnostic mammo labels." << std::endl;


    this->m_ImDiagnosticLabels = this->GenerateRegionLabels( this->m_BreastSideDiagnostic,
                                                 this->m_DiagTumourCenterIndex,
                                                 this->m_DiagTumourRegion,
                                                 this->m_DiagTumourRegionValue,
                                                 this->m_ImDiagnostic,
                                                 this->m_ImDiagnosticMask,
                                                 this->m_DiagPatches,
                                                 this->m_ThresholdDiagnostic );

    if ( this->m_FlgDebug )
    {
      this->template WriteImageFile<LabelImageType>( this->m_FileDiagnostic,
                                      std::string( "_DiagLabels.dcm" ),
                                      "diagnostic labels",
                                      this->m_ImDiagnosticLabels, this->m_DiagDictionary );
    }

    this->WriteLabelImageFile( this->m_FileDiagnostic,
                         std::string( "_DiagLabels.jpg" ),
                         "diagnostic labels",
                         this->m_ImDiagnosticLabels,  this->m_DiagTumourRegion,
                         this->m_DiagDictionary );
  }

  // Calculate the pre-diagnostic labels

  if ( this->m_ImPreDiagnostic )
  {
    if ( this->m_FlgVerbose )
      std::cout << "Computing pre-diagnostic mammo labels." << std::endl;

    if ( this->m_FlgRegister || this->m_FlgRegisterNonRigid )
    {
<<<<<<< HEAD
      this->m_PreDiagCenterIndex = this->TransformTumourPositionIntoImage( this->m_DiagTumourCenterIndex,
                                                               this->m_ImPreDiagnostic,
                                                               this->m_RegistrationPreDiag );
=======
      std::cout << "This is a right mammogram so flipping the tumour location in 'x'"
                << std::endl;
    }

    m_TumourLeft = imSizeInPixels[0] - m_TumourLeft;
    m_TumourRight = imSizeInPixels[0] - m_TumourRight;
  }

>>>>>>> Issue #4270: Modification of RegionalMammographicDensity to use MammogramRegistrationFilter

      if ( this->m_FlgVerbose )
        std::cout << "   Tumour center in pre-diag image: "
                  << this->m_PreDiagCenterIndex[0] << ", "
                  << this->m_PreDiagCenterIndex[1] << std::endl;
    }
    else
    {
      this->GenerateRandomTumourPositionInImage( Superclass::PREDIAGNOSTIC_MAMMO );
    }

    this->m_ImPreDiagnosticLabels = this->GenerateRegionLabels( this->m_BreastSidePreDiagnostic,
                                                    this->m_PreDiagCenterIndex,
                                                    this->m_PreDiagTumourRegion,
                                                    this->m_PreDiagTumourRegionValue,
                                                    this->m_ImPreDiagnostic,
                                                    this->m_ImPreDiagnosticMask,
                                                    this->m_PreDiagPatches,
                                                    this->m_ThresholdPreDiagnostic );

    if ( this->m_FlgDebug )
    {
      this->template WriteImageFile<LabelImageType>( this->m_FilePreDiagnostic,
                                      std::string( "_PreDiagLabels.dcm" ),
                                      "pre-diagnostic labels",
                                      this->m_ImPreDiagnosticLabels, this->m_PreDiagDictionary );
    }

    this->WriteLabelImageFile( this->m_FilePreDiagnostic,
                         std::string( "_PreDiagLabels.jpg" ),
                         "pre-diagnostic labels",
                         this->m_ImPreDiagnosticLabels, this->m_PreDiagTumourRegion,
                         this->m_PreDiagDictionary );
  }

  // Calculate the control-diagnostic labels

  if ( this->m_ImControl )
  {
    if ( this->m_FlgVerbose )
      std::cout << "Computing control mammo labels." << std::endl;

    if ( this->m_FlgRegister || this->m_FlgRegisterNonRigid )
    {
      this->m_ControlCenterIndex = this->TransformTumourPositionIntoImage( this->m_DiagTumourCenterIndex,
									   this->m_ImControl,
									   this->m_RegistrationControl );

      if ( this->m_FlgVerbose )
        std::cout << "   Tumour center in control image: "
                  << this->m_ControlCenterIndex[0] << ", "
                  << this->m_ControlCenterIndex[1] << std::endl;
    }
    else
    {
      this->GenerateRandomTumourPositionInImage( Superclass::CONTROL_MAMMO );
    }

    this->m_ImControlLabels = this->GenerateRegionLabels( this->m_BreastSideControl,
							  this->m_ControlCenterIndex,
							  this->m_ControlTumourRegion,
							  this->m_ControlTumourRegionValue,
							  this->m_ImControl,
							  this->m_ImControlMask,
							  this->m_ControlPatches,
							  this->m_ThresholdControl );

<<<<<<< HEAD
    if ( this->m_FlgDebug )
    {
      this->template WriteImageFile<LabelImageType>( this->m_FileControl,
						     std::string( "_ControlLabels.dcm" ),
						     "control labels",
						     this->m_ImControlLabels,
						     this->m_ControlDictionary );
    }

    this->WriteLabelImageFile( this->m_FileControl,
			       std::string( "_ControlLabels.jpg" ),
			       "control labels",
			       this->m_ImControlLabels, this->m_ControlTumourRegion,
			       this->m_ControlDictionary );
  }
=======
    // And registration version?

    if ( m_FlgRegister || m_FlgRegisterNonRigid )
    {
      m_FileDiagnosticRegnMask = BuildOutputFilename( m_FileDiagnostic, diagRegnMaskSuffix );

      if ( niftk::FileExists( m_FileDiagnosticRegnMask ) )
      {
        reader->SetFileName( m_FileDiagnosticRegnMask );

        try
        {
          std::cout << "Reading the diagnostic registration mask: " << m_FileDiagnosticRegnMask << std::endl;
          reader->Update();
        }

        catch (ExceptionObject &ex)
        {
          std::cerr << "ERROR: Could not read file: " 
                    << m_FileDiagnosticRegnMask << std::endl << ex << std::endl;
          throw( ex );
        }
        
        m_ImDiagnosticRegnMask = reader->GetOutput();
        m_ImDiagnosticRegnMask->DisconnectPipeline();
      }
      else
      {
        itkExceptionMacro( << "ERROR: Cannot read diagnostic registration mask: " << m_FileDiagnosticRegnMask );
      }
    }
  }
  else
  {
    m_ImDiagnosticMask = MaskWithPolygon( DIAGNOSTIC_MAMMO );

    CastImageAndWriteToFile< unsigned char >( m_FileDiagnostic, 
                                              diagMaskSuffix, 
                                              "diagnostic mask", 
                                              m_ImDiagnosticMask, 
                                              m_DiagDictionary );
  }


  // Generate the Pre-diagnostic Mask

  fileMask = BuildOutputFilename( m_FilePreDiagnostic, preDiagMaskSuffix );

  if ( niftk::FileExists( fileMask ) && ( ! m_FlgOverwrite ) )
  {
    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( fileMask );

    try
    {
      std::cout << "Reading the pre-diagnostic mask: " << fileMask << std::endl;
      reader->Update();
    }

    catch (ExceptionObject &ex)
    {
      std::cerr << "ERROR: Could not read file: " 
                << fileMask << std::endl << ex << std::endl;
      throw( ex );
    }

    m_ImPreDiagnosticMask = reader->GetOutput();
    m_ImPreDiagnosticMask->DisconnectPipeline();

    // And registration version?

    if ( m_FlgRegister || m_FlgRegisterNonRigid )
    {
      fileRegnMask = BuildOutputFilename( m_FilePreDiagnostic, preDiagRegnMaskSuffix );

      if ( niftk::FileExists( fileRegnMask ) )
      {
        reader->SetFileName( fileRegnMask );

        try
        {
          std::cout << "Reading the pre-diagnostic registration mask: " << fileRegnMask << std::endl;
          reader->Update();
        }

        catch (ExceptionObject &ex)
        {
          std::cerr << "ERROR: Could not read file: " 
                    << fileRegnMask << std::endl << ex << std::endl;
          throw( ex );
        }
        
        m_ImPreDiagnosticRegnMask = reader->GetOutput();
        m_ImPreDiagnosticRegnMask->DisconnectPipeline();
      }
      else
      {
        itkExceptionMacro( << "ERROR: Cannot read pre-diagnostic registration mask: " << fileRegnMask );
      }
    }
  }
  else
  {
    m_ImPreDiagnosticMask = MaskWithPolygon( PREDIAGNOSTIC_MAMMO );

    CastImageAndWriteToFile< unsigned char >( m_FilePreDiagnostic, 
                                              preDiagMaskSuffix, 
                                              "pre-diagnostic mask", 
                                              m_ImPreDiagnosticMask, 
                                              m_PreDiagDictionary );
  }

  // Generate the Control Mask

  fileMask = BuildOutputFilename( m_FileControl, controlMaskSuffix );

  if ( niftk::FileExists( fileMask ) && ( ! m_FlgOverwrite ) )
  {
    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( fileMask );

    try
    {
      std::cout << "Reading the control mask: " << fileMask << std::endl;
      reader->Update();
    }

    catch (ExceptionObject &ex)
    {
      std::cerr << "ERROR: Could not read file: " 
                << fileMask << std::endl << ex << std::endl;
      throw( ex );
    }

    m_ImControlMask = reader->GetOutput();
    m_ImControlMask->DisconnectPipeline();

    // And registration version?

    if ( m_FlgRegister || m_FlgRegisterNonRigid )
    {
      fileRegnMask = BuildOutputFilename( m_FileControl, controlRegnMaskSuffix );

      if ( niftk::FileExists( fileRegnMask ) )
      {
        reader->SetFileName( fileRegnMask );

        try
        {
          std::cout << "Reading the control registration mask: " << fileRegnMask << std::endl;
          reader->Update();
        }

        catch (ExceptionObject &ex)
        {
          std::cerr << "ERROR: Could not read file: " 
                    << fileRegnMask << std::endl << ex << std::endl;
          throw( ex );
        }
        
        m_ImControlRegnMask = reader->GetOutput();
        m_ImControlRegnMask->DisconnectPipeline();
      }
      else
      {
        itkExceptionMacro( << "ERROR: Cannot read control registration mask: " << fileRegnMask );
      }
    }
  }
  else
  {
    m_ImControlMask = MaskWithPolygon( CONTROL_MAMMO );

    CastImageAndWriteToFile< unsigned char >( m_FileControl, 
                                              controlMaskSuffix, 
                                              "control mask", 
                                              m_ImControlMask, 
                                              m_ControlDictionary );
  }


  // Register the images?

  if ( m_FlgRegister )
  {
    RegisterTheImages();
  }


  // Calculate the diagnostic labels 

  if ( m_FlgVerbose ) 
    std::cout << "Computing diagnostic mammo labels." << std::endl;


  m_ImDiagnosticLabels = GenerateRegionLabels( m_BreastSideDiagnostic,
                                               m_DiagTumourCenterIndex,
                                               m_DiagTumourRegion,
                                               m_DiagTumourRegionValue,
                                               m_ImDiagnostic, 
                                               m_ImDiagnosticMask, 
                                               m_DiagPatches, 
                                               m_ThresholdDiagnostic );

  if ( m_FlgDebug )
  {
    WriteImageFile<LabelImageType>( m_FileDiagnostic, 
                                    std::string( "_DiagLabels.dcm" ), 
                                    "diagnostic labels", 
                                    m_ImDiagnosticLabels, m_DiagDictionary );
  }

  WriteLabelImageFile( m_FileDiagnostic, 
                       std::string( "_DiagLabels.jpg" ), 
                       "diagnostic labels", 
                       m_ImDiagnosticLabels,  m_DiagTumourRegion,
                       m_DiagDictionary );
  
  // Calculate the pre-diagnostic labels 

  if ( m_FlgVerbose ) 
    std::cout << "Computing pre-diagnostic mammo labels." << std::endl;

  if ( m_FlgRegister || m_FlgRegisterNonRigid ) 
  {
    m_PreDiagCenterIndex = TransformTumourPositionIntoImage( m_DiagTumourCenterIndex,
                                                             m_ImPreDiagnostic,
                                                             m_RegistrationPreDiag );

    if ( m_FlgVerbose ) 
      std::cout << "   Tumour center in pre-diag image: " 
                << m_PreDiagCenterIndex[0] << ", " 
                << m_PreDiagCenterIndex[1] << std::endl;    
  }
  else 
  {
    GenerateRandomTumourPositionInImage( PREDIAGNOSTIC_MAMMO );
  }

  m_ImPreDiagnosticLabels = GenerateRegionLabels( m_BreastSidePreDiagnostic,
                                                  m_PreDiagCenterIndex,
                                                  m_PreDiagTumourRegion,
                                                  m_PreDiagTumourRegionValue,
                                                  m_ImPreDiagnostic, 
                                                  m_ImPreDiagnosticMask, 
                                                  m_PreDiagPatches,
                                                  m_ThresholdPreDiagnostic );

  if ( m_FlgDebug )
  {
    WriteImageFile<LabelImageType>( m_FilePreDiagnostic, 
                                    std::string( "_PreDiagLabels.dcm" ), 
                                    "pre-diagnostic labels", 
                                    m_ImPreDiagnosticLabels, m_PreDiagDictionary );
  }

  WriteLabelImageFile( m_FilePreDiagnostic, 
                       std::string( "_PreDiagLabels.jpg" ), 
                       "pre-diagnostic labels", 
                       m_ImPreDiagnosticLabels, m_PreDiagTumourRegion,
                       m_PreDiagDictionary );
  
  
  // Calculate the control-diagnostic labels 

  if ( m_FlgVerbose ) 
    std::cout << "Computing control mammo labels." << std::endl;

  if ( m_FlgRegister || m_FlgRegisterNonRigid ) 
  {
    m_ControlCenterIndex = TransformTumourPositionIntoImage( m_DiagTumourCenterIndex,
                                                             m_ImControl,
                                                             m_RegistrationControl );

    if ( m_FlgVerbose ) 
      std::cout << "   Tumour center in control image: " 
                << m_ControlCenterIndex[0] << ", " 
                << m_ControlCenterIndex[1] << std::endl;    
  }
  else 
  {
    GenerateRandomTumourPositionInImage( CONTROL_MAMMO );
  }

  m_ImControlLabels = GenerateRegionLabels( m_BreastSideControl,
                                            m_ControlCenterIndex,
                                            m_ControlTumourRegion,
                                            m_ControlTumourRegionValue,
                                            m_ImControl, 
                                            m_ImControlMask, 
                                            m_ControlPatches,
                                            m_ThresholdControl );

  if ( m_FlgDebug )
  {
    WriteImageFile<LabelImageType>( m_FileControl, 
                                    std::string( "_ControlLabels.dcm" ), 
                                    "control labels", 
                                    m_ImControlLabels, m_ControlDictionary );
  }

  WriteLabelImageFile( m_FileControl, 
                       std::string( "_ControlLabels.jpg" ), 
                       "control labels", 
                       m_ImControlLabels, m_ControlTumourRegion,
                       m_ControlDictionary );
  
};


// --------------------------------------------------------------------------
// BuildOutputFilename()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
std::string
RegionalMammographicDensity< InputPixelType, InputDimension >
::BuildOutputFilename( std::string fileInput, std::string suffix )
{
  std::string fileOutput, dirOutputFullPath;

  fileOutput = niftk::ModifyImageFileSuffix( fileInput, suffix );

  fileOutput = niftk::ConcatenatePath( m_DirOutput, fileOutput );

  dirOutputFullPath = fs::path( fileOutput ).branch_path().string();
  
  if ( ! niftk::DirectoryExists( dirOutputFullPath ) )
  {
    std::cout << "Creating output directory: " << dirOutputFullPath << std::endl;
    niftk::CreateDirAndParents( dirOutputFullPath );
  }
    
  std::cout << "Output filename: " << fileOutput << std::endl;

  return fileOutput;
}


// --------------------------------------------------------------------------
// CastImageAndWriteToFile()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
template <typename TOutputImageType>
std::string
RegionalMammographicDensity< InputPixelType, InputDimension >
::CastImageAndWriteToFile( std::string fileInput, 
                            std::string suffix,
                            const char *description,
                            typename ImageType::Pointer image,
                            DictionaryType &dictionary )
{
  std::string fileModifiedOutput;

  if ( fileInput.length() ) 
  {
    typedef RescaleIntensityImageFilter< ImageType, OutputImageType > CastFilterType;
    typedef ImageFileWriter< OutputImageType > FileWriterType;

    typename ImageType::Pointer pipeITKImageDataConnector;

    typename CastFilterType::Pointer caster = CastFilterType::New();

    fileModifiedOutput = BuildOutputFilename( fileInput, suffix );

    if ( niftk::FileExists( fileModifiedOutput ) ) 
    {
      if ( ! m_FlgOverwrite ) 
      {
        std::cerr << std::endl << "WARNING: File " << fileModifiedOutput << " exists"
                  << std::endl << "         and can't be overwritten. Consider option: 'overwrite'."
                  << std::endl << std::endl;
        return fileModifiedOutput;
      }
      else
      {
        std::cerr << std::endl << "WARNING: File " << fileModifiedOutput << " exists"
                  << std::endl << "         and will be overwritten."
                  << std::endl << std::endl;
      }
    }

    caster->SetInput( image );
    caster->SetOutputMinimum(   0 );
    caster->SetOutputMaximum( 255 );

    try
    {
      caster->UpdateLargestPossibleRegion();
    }    
    catch (ExceptionObject &ex)
    {
      std::cerr << ex << std::endl;
      throw( ex );
    }

    typename FileWriterType::Pointer writer = FileWriterType::New();

    typename OutputImageType::Pointer outImage = caster->GetOutput();

    outImage->DisconnectPipeline();

    typename ImageIOBase::Pointer imageIO;
    imageIO = ImageIOFactory::CreateImageIO(fileModifiedOutput.c_str(), 
                                            ImageIOFactory::WriteMode);

    imageIO->SetMetaDataDictionary( dictionary );

    writer->SetFileName( fileModifiedOutput.c_str() );
    writer->SetInput( outImage );
    writer->SetImageIO( imageIO );
    writer->UseInputMetaDataDictionaryOff();

    try
    {
      std::cout << "Writing " << description << " to file: "
                << fileModifiedOutput.c_str() << std::endl;
      writer->Update();
    }
  
    catch (ExceptionObject &ex)
    {
      std::cerr << "ERROR: Could not write file: " << fileModifiedOutput << std::endl 
                << ex << std::endl;
      throw( ex );
    }
  }
  else
  {
    std::cerr << "Failed to write " << description 
              << " to file - filename is empty " << std::endl;
  }

  return fileModifiedOutput;
}


// --------------------------------------------------------------------------
// WriteImageFile()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
template <typename TOutputImageType>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::WriteImageFile( std::string fileInput, 
                  std::string suffix, 
                  const char *description,
                  typename TOutputImageType::Pointer image,
                  DictionaryType &dictionary )
{
  if ( fileInput.length() ) 
  {
    typedef ImageFileWriter< TOutputImageType > FileWriterType;

    std::string fileModifiedOutput = BuildOutputFilename( fileInput, suffix );

    if ( niftk::FileExists( fileModifiedOutput ) ) 
    {
      std::cerr << std::endl << "WARNING: File " << fileModifiedOutput << " exists"
                << std::endl << "         but will be overwritten."
                << std::endl << std::endl;
    }

    typename FileWriterType::Pointer writer = FileWriterType::New();

    image->DisconnectPipeline();

    typename ImageIOBase::Pointer imageIO;
    imageIO = ImageIOFactory::CreateImageIO(fileModifiedOutput.c_str(), 
                                            ImageIOFactory::WriteMode);

    imageIO->SetMetaDataDictionary( dictionary );

    writer->SetFileName( fileModifiedOutput.c_str() );
    writer->SetInput( image );
    writer->SetImageIO( imageIO );
    writer->UseInputMetaDataDictionaryOff();

    try
    {
      std::cout << "Writing " << description << " to file: "
                << fileModifiedOutput.c_str() << std::endl;
      writer->Update();
    }
  
    catch (ExceptionObject &ex)
    {
      std::cerr << "ERROR: Could notwrite file: " << fileModifiedOutput << std::endl 
                << ex << std::endl;
      throw( ex );
    }
  }
  else
  {
    itkExceptionMacro( << "Failed to write " << description 
                       << " to file - filename is empty " );
  }
}


// --------------------------------------------------------------------------
// WriteLabelImageFile()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::WriteLabelImageFile( std::string fileInput, 
                       std::string suffix, 
                       const char *description,
                       typename LabelImageType::Pointer image,
                       typename LabelImageType::RegionType &tumourRegion,
                       DictionaryType &dictionary )
{
  if ( fileInput.length() ) 
  {

    typedef itk::RGBPixel<unsigned char>             RGBPixelType;
    typedef itk::Image<RGBPixelType, InputDimension> RGBImageType;
 
#if 0 
    LabelPixelType distanceThreshold = 0;
 
    typedef itk::ScalarConnectedComponentImageFilter <LabelImageType, LabelImageType >
      ConnectedComponentImageFilterType;
 
    typename ConnectedComponentImageFilterType::Pointer connected =
      ConnectedComponentImageFilterType::New ();

    connected->SetInput(image);
    connected->SetDistanceThreshold(distanceThreshold);
 
    typedef itk::RelabelComponentImageFilter <LabelImageType, LabelImageType >
      RelabelFilterType;

    typename RelabelFilterType::Pointer relabel = RelabelFilterType::New();
    typename RelabelFilterType::ObjectSizeType minSize = 20;

    relabel->SetInput(connected->GetOutput());
    relabel->SetMinimumObjectSize(minSize);

    try {
      std::cout << "Computing connected labels" << std::endl;
      relabel->Update();
    }
    catch (ExceptionObject &ex)
    {
      std::cerr << ex << std::endl;
      throw( ex );
    }
#endif

    // Create a color image and set the background to black

    typedef itk::LabelToRGBImageFilter<LabelImageType, RGBImageType> RGBFilterType;

    typename RGBFilterType::Pointer rgbFilter = RGBFilterType::New();

    //rgbFilter->SetInput( relabel->GetOutput() );
    rgbFilter->SetInput( image );

    RGBPixelType rgbPixel;
    
    rgbPixel.SetRed(   0 );
    rgbPixel.SetGreen( 0 );
    rgbPixel.SetBlue(  0 );
    
    rgbFilter->SetBackgroundValue( 0 );
    rgbFilter->SetBackgroundColor( rgbPixel );
 
    try {
      rgbFilter->Update();
    }
    catch (ExceptionObject &ex)
    {
      std::cerr << ex << std::endl;
      throw( ex );
    }

    typename RGBImageType::Pointer rgbImage = rgbFilter->GetOutput();
    rgbImage->DisconnectPipeline();
  

    // Set the tumour region to white

    itk::ImageRegionIterator< LabelImageType > itLabel( image, tumourRegion );
    itk::ImageRegionIterator< RGBImageType > itRGBImage( rgbImage, tumourRegion );
      
    rgbPixel.SetRed(   255 );
    rgbPixel.SetGreen( 255 );
    rgbPixel.SetBlue(  255 );
    
    for ( itRGBImage.GoToBegin(), itLabel.GoToBegin(); 
          ! itRGBImage.IsAtEnd(); 
          ++itRGBImage, ++itLabel )
    {
      if ( itLabel.Get() )
      {
        itRGBImage.Set( rgbPixel );
      }
    }
   

    // Write the output to a file

    typedef ImageFileWriter< RGBImageType > FileWriterType;

    std::string fileModifiedOutput = BuildOutputFilename( fileInput, suffix );

    if ( niftk::FileExists( fileModifiedOutput ) ) 
    {
      std::cerr << std::endl << "WARNING: File " << fileModifiedOutput << " exists"
                << std::endl << "         but will be overwritten."
                << std::endl << std::endl;
    }

    typename FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( fileModifiedOutput.c_str() );
    writer->SetInput( rgbImage );

    try
    {
      std::cout << "Writing " << description << " to file: "
                << fileModifiedOutput.c_str() << std::endl;
      writer->Update();
    }
  
    catch (ExceptionObject &ex)
    {
      std::cerr << "ERROR: Could not write file: " << fileModifiedOutput << std::endl 
                << ex << std::endl;
      throw( ex );
    }
  }
  else
  {
    itkExceptionMacro( << "Failed to write " << description 
                       << " to file - filename is empty " );
  }
}


// --------------------------------------------------------------------------
// AddPointToPolygon()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::AddPointToPolygon( typename PolygonType::Pointer &polygon, 
                     typename ImageType::Pointer &image, 
                     typename ImageType::SizeType &polySize, 
                     int x, int y )
{
  typename ImageType::IndexType index;
  typename PolygonType::PointType point;
  typename PolygonType::PointListType points = polygon->GetPoints();

  index[0] = x;
  index[1] = y;

  if ( x > polySize[0] ) polySize[0] = x;
  if ( y > polySize[1] ) polySize[1] = y;

  image->TransformIndexToPhysicalPoint( index, point );
  
  if ( (points.size() == 0) ||
       ( ( point[0] != points.back().GetPosition()[0] ) || 
         ( point[1] != points.back().GetPosition()[1] ) ) )
  {
    polygon->AddPoint( point );
  }
}


// --------------------------------------------------------------------------
// MaskWithPolygon()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
typename RegionalMammographicDensity< InputPixelType, InputDimension >::ImageType::Pointer 
RegionalMammographicDensity< InputPixelType, InputDimension >
::MaskWithPolygon( MammogramType mammoType, LocusType locusType )
{
  typename ImageType::Pointer imMask;
  typename ImageType::Pointer image;    
  typename ImageType::RegionType region;
  typename ImageType::SizeType polySize;

  typename std::vector< PointOnBoundary > *pPointOnBoundary;

  std::vector< PointOnBoundary >::iterator itPointOnBoundary;

  polySize[0] = 0;
  polySize[1] = 0;


  // Set the image and loci
   
  if ( mammoType == DIAGNOSTIC_MAMMO )
  {
    image = m_ImDiagnostic;

    if ( locusType == BREAST_EDGE )
    {
      if ( m_FlgVerbose ) 
        std::cout << "Creating diagnostic mammo breast edge mask." << std::endl;

      pPointOnBoundary = &m_DiagBreastEdgePoints; 
    }
    else
    {
      if ( m_FlgVerbose ) 
        std::cout << "Creating diagnostic mammo pectoral mask." << std::endl;

      pPointOnBoundary = &m_DiagPectoralPoints;
    }
  }
  else if ( mammoType == PREDIAGNOSTIC_MAMMO )
  {
    image = m_ImPreDiagnostic;

    if ( locusType == BREAST_EDGE )
    {
      if ( m_FlgVerbose ) 
        std::cout << "Creating prediagnostic mammo breast edge mask." << std::endl;

      pPointOnBoundary = &m_PreDiagBreastEdgePoints; 
    }
    else {
      if ( m_FlgVerbose ) 
        std::cout << "Creating prediagnostic mammo pectoral mask." << std::endl;

      pPointOnBoundary = &m_PreDiagPectoralPoints;
    }
  }
  else if ( mammoType == CONTROL_MAMMO )
  {
    image = m_ImControl;

    if ( locusType == BREAST_EDGE )
    {
      if ( m_FlgVerbose ) 
        std::cout << "Creating control mammo breast edge mask." << std::endl;

      pPointOnBoundary = &m_ControlBreastEdgePoints; 
    }
    else {
      if ( m_FlgVerbose ) 
        std::cout << "Creating control mammo pectoral mask." << std::endl;

      pPointOnBoundary = &m_ControlPectoralPoints;
    }
  }

  if ( ! pPointOnBoundary )
  {
    itkExceptionMacro( << "ERROR: No boundary points defined" );
    return 0;
  }

  region = image->GetLargestPossibleRegion();


  // Sort the coordinates and check that the first point is closer to y = 0

  std::sort( pPointOnBoundary->begin(), pPointOnBoundary->end(), CmpCoordsAscending );

  pPointOnBoundary->front().Print( "  First point: " );
  pPointOnBoundary->back().Print(  "  Last point:  " );

  if ( pPointOnBoundary->front().y > pPointOnBoundary->back().y )
  {
    std::sort( pPointOnBoundary->begin(), pPointOnBoundary->end(), CmpCoordsDescending );

    pPointOnBoundary->front().Print( "  First point: " );
    pPointOnBoundary->back().Print(  "  Last point:  " );
  }

  typename PolygonType::Pointer polygon = PolygonType::New();

 
  polygon->ComputeObjectToWorldTransform();

  // First point is (x=0, y=0)   
  AddPointToPolygon( polygon, image, polySize, 0, 0 );

  // Second point is (x, y=0)   
  AddPointToPolygon( polygon, image, polySize, pPointOnBoundary->front().x, 0 );

  // Add all the user specified points
  for ( itPointOnBoundary  = pPointOnBoundary->begin(); 
        itPointOnBoundary != pPointOnBoundary->end(); 
        itPointOnBoundary++ )
  {
    (*itPointOnBoundary).Print( "     " );

    AddPointToPolygon( polygon, image, polySize, (*itPointOnBoundary).x, (*itPointOnBoundary).y );
  }

  // Last point is (x=0, y) for breast edge
  if ( locusType == BREAST_EDGE )
  {
    AddPointToPolygon( polygon, image, polySize, 0, pPointOnBoundary->back().y );
  }
  // or (x=0, y=max) for pectoral
  else {
    AddPointToPolygon( polygon, image, polySize, 0, region.GetSize()[1] );
  }

  if ( m_FlgDebug )
  {
    typename PolygonType::PointListType vertexList = polygon->GetPoints();
    typename PolygonType::PointListType::const_iterator itPoints = vertexList.begin();

    std::cout << "  Polygon vertices:" << std::endl;

    for (itPoints=vertexList.begin(); itPoints<vertexList.end(); itPoints++)
    {
      std::cout << "     " 
                << std::right << std::setw(8) << (*itPoints).GetPosition()[0] << ", " 
                << std::right << std::setw(8) << (*itPoints).GetPosition()[1] << std::endl;
    }
  }

  // Create the mask

  typedef SpatialObjectToImageFilter< PolygonType, 
    ImageType > SpatialObjectToImageFilterType;

  typename SpatialObjectToImageFilterType::Pointer 
    polyMaskFilter = SpatialObjectToImageFilterType::New();
 
  polyMaskFilter->SetInput( polygon );
  polyMaskFilter->SetInsideValue( 1000 );
  polyMaskFilter->SetOutsideValue( 0 );
    
  polyMaskFilter->SetSize( region.GetSize() );
  polyMaskFilter->SetSpacing( image->GetSpacing() );

  polygon->SetThickness(1.0);

  try
  {
    polyMaskFilter->Update();
  }
  
  catch (ExceptionObject &ex)
  {
    std::cerr << ex << std::endl;
    throw( ex );
  }

  imMask = polyMaskFilter->GetOutput();
  imMask->DisconnectPipeline();

  imMask->SetSpacing( image->GetSpacing() );


  // If this is a right mammogram then flip it

  if ( ( ( mammoType == DIAGNOSTIC_MAMMO ) && 
         ( m_BreastSideDiagnostic    == LeftOrRightSideCalculatorType::RIGHT_BREAST_SIDE ) ) || 
       ( ( mammoType == PREDIAGNOSTIC_MAMMO ) && 
         ( m_BreastSidePreDiagnostic == LeftOrRightSideCalculatorType::RIGHT_BREAST_SIDE ) ) || 
       ( ( mammoType == CONTROL_MAMMO ) && 
         ( m_BreastSideControl       == LeftOrRightSideCalculatorType::RIGHT_BREAST_SIDE ) ) )
  {
    itk::FixedArray<bool, 2> flipAxes;
    flipAxes[0] = true;
    flipAxes[1] = false;

    if ( m_FlgDebug )
    {
      std::cout << "This is a right mammogram so flipping the mask in 'x'" << std::endl;
    }

    typename ImageType::PointType origin;
    origin = imMask->GetOrigin();

    typedef itk::FlipImageFilter< ImageType > FlipImageFilterType;
 
    typename FlipImageFilterType::Pointer flipFilter = FlipImageFilterType::New ();

    flipFilter->SetInput( imMask );
    flipFilter->SetFlipAxes( flipAxes );

    try
    {
      flipFilter->Update();
    }
    
    catch (ExceptionObject &ex)
    {
      std::cerr << ex << std::endl;
      throw( ex );
    }
 
    imMask = flipFilter->GetOutput();
    imMask->DisconnectPipeline();

    imMask->SetOrigin( origin );
  }

  return imMask;
}


// --------------------------------------------------------------------------
// MaskWithPolygon()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
typename RegionalMammographicDensity< InputPixelType, InputDimension >::ImageType::Pointer 
RegionalMammographicDensity< InputPixelType, InputDimension >
::MaskWithPolygon( MammogramType mammoType )
{
  typename ImageType::Pointer imMask, imMaskPec;

  imMask = MaskWithPolygon( mammoType, BREAST_EDGE );
  imMaskPec = MaskWithPolygon( mammoType, PECTORAL );

  std::cout << "Breast Edge mask spacing: " << imMask->GetSpacing() << std::endl;
  std::cout << "Pectoral mask spacing: " << imMaskPec->GetSpacing() << std::endl;

  if ( m_FlgDebug ) 
  {
    
    if ( mammoType == DIAGNOSTIC_MAMMO )
    {
      CastImageAndWriteToFile< unsigned char >( m_FileDiagnostic, 
                                                std::string( "_DiagBreastEdgeMask.dcm" ), 
                                                "diagnostic breast edge mask", 
                                                imMask, m_DiagDictionary );
    }
    else if ( mammoType == PREDIAGNOSTIC_MAMMO )
    {
      CastImageAndWriteToFile< unsigned char >( m_FilePreDiagnostic, 
                                                std::string( "_PreDiagBreastEdgeMask.dcm" ), 
                                                "pre-diagnostic breast edge mask", 
                                                imMask, m_PreDiagDictionary );
    }
    else
    {
      CastImageAndWriteToFile< unsigned char >( m_FileControl, 
                                                std::string( "_ControlBreastEdgeMask.dcm" ), 
                                                "control breast edge mask", 
                                                imMask, m_ControlDictionary );
    }

    if ( mammoType == DIAGNOSTIC_MAMMO )
    {
      CastImageAndWriteToFile< unsigned char >( m_FileDiagnostic, 
                                                std::string( "_DiagPectoralMask.dcm" ), 
                                                "diagnostic pectoral mask",
                                                imMaskPec, m_DiagDictionary );
    }
    else if ( mammoType == PREDIAGNOSTIC_MAMMO )
    {
      CastImageAndWriteToFile< unsigned char >( m_FilePreDiagnostic, 
                                                std::string( "_PreDiagPectoralMask.dcm" ), 
                                                "pre-diagnostic pectoral mask", 
                                                imMaskPec, m_PreDiagDictionary );
    }
    else
    {
      CastImageAndWriteToFile< unsigned char >( m_FileControl, 
                                                std::string( "_ControlPectoralMask.dcm" ), 
                                                "control pectoral mask", 
                                                imMaskPec, m_ControlDictionary );
    }
  }

  // If we're registering the images then expand the breast edge to use as a mask
  // by thresholding a distance transform of the edge mask at 10mm

  if ( m_FlgRegister || m_FlgRegisterNonRigid )
  {

    typename DistanceTransformType::Pointer distanceTransform = DistanceTransformType::New();

    distanceTransform->SetInput( imMask );
    distanceTransform->SetInsideIsPositive( false );
    distanceTransform->UseImageSpacingOn();
    distanceTransform->SquaredDistanceOff();

    try
    {
      std::cout << "Computing distance transform of breast edge mask for registration" << std::endl;
      distanceTransform->UpdateLargestPossibleRegion();
    }    
    catch (ExceptionObject &ex)
    {
      std::cerr << ex << std::endl;
      throw( ex );
    }

    typedef typename itk::BinaryThresholdImageFilter<RealImageType, ImageType> BinaryThresholdFilterType;
  
    
    typename BinaryThresholdFilterType::Pointer thresholdFilter = BinaryThresholdFilterType::New();

    RealType threshold = 10;
    
    thresholdFilter->SetInput( distanceTransform->GetOutput() );
    
    thresholdFilter->SetOutsideValue( 0 );
    thresholdFilter->SetInsideValue( 255 );
    thresholdFilter->SetLowerThreshold( threshold );
    
    try
    {
      std::cout << "Thresholding distance transform of breast edge mask at: "
                << threshold << std::endl;
      thresholdFilter->UpdateLargestPossibleRegion();
    }    
    catch (ExceptionObject &ex)
    {
      std::cerr << ex << std::endl;
      throw( ex );
    }

    typedef typename itk::InvertIntensityBetweenMaxAndMinImageFilter<ImageType> InvertFilterType;

    typename InvertFilterType::Pointer invertFilter = InvertFilterType::New();
    invertFilter->SetInput( thresholdFilter->GetOutput() );
    
    try
    {
      std::cout << "Inverting the registration mask" << std::endl;
      invertFilter->UpdateLargestPossibleRegion();
    }    
    catch (ExceptionObject &ex)
    {
      std::cerr << ex << std::endl;
      throw( ex );
    }
    
    if ( mammoType == DIAGNOSTIC_MAMMO )
    {
      m_ImDiagnosticRegnMask = invertFilter->GetOutput();
      m_ImDiagnosticRegnMask->DisconnectPipeline();
      
      RemoveTumourFromRegnMask();
      
      m_FileDiagnosticRegnMask = 
        CastImageAndWriteToFile< unsigned char >( m_FileDiagnostic, 
                                                  std::string( "_DiagRegnMask.dcm" ), 
                                                  "diagnostic pectoral registration mask",
                                                  m_ImDiagnosticRegnMask, m_DiagDictionary );
    }
    else if ( mammoType == PREDIAGNOSTIC_MAMMO )
    {
      m_ImPreDiagnosticRegnMask = invertFilter->GetOutput();
      m_ImPreDiagnosticRegnMask->DisconnectPipeline();
      
      CastImageAndWriteToFile< unsigned char >( m_FilePreDiagnostic, 
                                                std::string( "_PreDiagRegnMask.dcm" ), 
                                                "pre-diagnostic registration mask", 
                                                m_ImPreDiagnosticRegnMask, m_PreDiagDictionary );
    }
    else
    {
      m_ImControlRegnMask = invertFilter->GetOutput();
      m_ImControlRegnMask->DisconnectPipeline();
      
      CastImageAndWriteToFile< unsigned char >( m_FileControl, 
                                                std::string( "_ControlRegnMask.dcm" ), 
                                                "control registration mask", 
                                                m_ImControlRegnMask, m_ControlDictionary );
    }
  }


  // Iterate over the masks to compute the combined mask
  
  IteratorType itMask( imMask, imMask->GetLargestPossibleRegion() );
  IteratorType itMaskPec( imMaskPec, imMaskPec->GetLargestPossibleRegion() );
      
  for ( itMask.GoToBegin(), itMaskPec.GoToBegin(); 
        ! itMask.IsAtEnd(); 
        ++itMask, ++itMaskPec )
  {
    if ( itMaskPec.Get() )
      itMask.Set( 0 );
  }

  return imMask;
}


// --------------------------------------------------------------------------
// RemoveTumourFromRegnMask()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::RemoveTumourFromRegnMask( void )
{
  if ( m_TumourDiameter == 0. )
  {
    return;
  }

  int i;
  double distToTumourCenter;

  typename ImageType::RegionType tumourRegion;
  typename ImageType::SizeType   tumourSize;

  typename ImageType::IndexType idxStart;
  typename ImageType::IndexType idxEnd;

  typename ImageType::PointType ptTumourCenter;
  typename ImageType::PointType pt;
  typename ImageType::PointType ptStart;
  typename ImageType::PointType ptEnd;

  m_ImDiagnosticRegnMask->TransformIndexToPhysicalPoint( m_DiagTumourCenterIndex, 
                                                         ptTumourCenter );


  for ( i=0; i<InputDimension; i++)
  {
    ptStart[i] = ptTumourCenter[i] - m_TumourDiameter/2.;
    ptEnd[i]   = ptTumourCenter[i] + m_TumourDiameter/2.;
  }

  m_ImDiagnosticRegnMask->TransformPhysicalPointToIndex( ptStart, idxStart );
  m_ImDiagnosticRegnMask->TransformPhysicalPointToIndex( ptEnd, idxEnd );
  
  tumourRegion.SetIndex( idxStart );

  for ( i=0; i<InputDimension; i++)
  {
    tumourSize[i] = idxEnd[i] - idxStart[i];
  }

  tumourRegion.SetSize( tumourSize );


  IteratorWithIndexType itMask( m_ImDiagnosticRegnMask, tumourRegion );
      
  for ( itMask.GoToBegin(); ! itMask.IsAtEnd(); ++itMask )
  {
    m_ImDiagnosticRegnMask->TransformIndexToPhysicalPoint( itMask.GetIndex(), pt );

    distToTumourCenter = sqrt( (pt[0] - ptTumourCenter[0])*(pt[0] - ptTumourCenter[0]) +
                               (pt[1] - ptTumourCenter[1])*(pt[1] - ptTumourCenter[1]) );

    if ( distToTumourCenter <= m_TumourDiameter/2. )
    {
      itMask.Set( 0 );
    }
  }
}


// --------------------------------------------------------------------------
// SetRegistrationParameterScales()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
template <typename ScalesType>
ScalesType
RegionalMammographicDensity< InputPixelType, InputDimension >
::SetRegistrationParameterScales( typename itk::TransformTypeEnum transformType,
                                  unsigned int nParameters )
{
  unsigned int i;
  ScalesType scales( nParameters);

  scales.Fill( 1.0 );      

  switch ( transformType )
  {

  // Translation only, 2 DOF in 2D and 3 DOF in 3D
  case TRANSLATION:
  {
    if ( nParameters != InputDimension )
    {
      itkExceptionMacro( << "ERROR: Number of registration parameters does not equal dimension" );
    }
    break;
  }

  // Rigid, so rotations and translations, 3 DOF in 2D and 6 DOF in 3D.
  case RIGID:
  {
    if (( InputDimension == 2) && (nParameters != 3) )
    {
      itkExceptionMacro( << "ERROR: Rigid transformation should have 3 parameters in 2D" );
    }
    else if (( InputDimension == 3) && (nParameters != 6) )
    {
      itkExceptionMacro( << "ERROR: Rigid transformation should have 6 parameters in 3D" );
    }
    break;
  }

  // Rigid plus scale, 5 DOF in 2D, 9 DOF in 3D.
  case RIGID_SCALE:
  {
    if ( InputDimension == 2 )
    {
      if ( nParameters != 5 )
      {
        itkExceptionMacro( << "ERROR: Rigid plus scaletransformation should have 5 parameters in 2D" );
      }

      for ( i=3; i<5; i++ )
      {
        scales[i] = 100.0; 
      }
    }
    else if ( InputDimension == 3 )
    {
      if ( nParameters != 9 )
      {
        itkExceptionMacro( << "ERROR: Rigid plus scale transformation should have 9 parameters in 3D" );
      }
      for ( i=6; i<9; i++ )
      {
        scales[i] = 100.0; 
      }
    }
    break;
  }

  // Affine. 6 DOF in 2D, 12 DOF in 3D.

  case AFFINE:
  {
    if ( InputDimension == 2 )
    {
      if ( nParameters != 6 )
      {
        itkExceptionMacro( << "ERROR: Affine transformation should have 6 parameters in 2D" );
      }

      for ( i=3; i<6; i++ )
      {
        scales[i] = 100.0; 
      }
    }
    else if ( InputDimension == 3 )
    {
      if ( nParameters != 12 )
      {
        itkExceptionMacro( << "ERROR: Affine transformation should have 12 parameters in 3D" );
      }
      for ( i=6; i<12; i++ )
      {
        scales[i] = 100.0; 
      }
    }
    break;
  }

  default: {
    itkExceptionMacro( << "ERROR: Unrecognised transformation type: " 
                       << transformType );
  }
  }

  return scales;
};


// --------------------------------------------------------------------------
// WriteRegistrationDifferenceImage()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::WriteRegistrationDifferenceImage( std::string fileInput, 
                                    std::string suffix, 
                                    const char *description,
                                    typename OutputImageType::Pointer image,
                                    DictionaryType &dictionary )
{
  if ( fileInput.length() ) 
  {
    typedef typename itk::CastImageFilter< ImageType, OutputImageType > CastFilterType;
    typename CastFilterType::Pointer castFilter = CastFilterType::New();
    castFilter->SetInput( m_ImDiagnostic );
 
    // Subtract the images

    typedef typename itk::SubtractImageFilter<OutputImageType, OutputImageType> SubtractFilterType;

    typename SubtractFilterType::Pointer subtractionFilter = SubtractFilterType::New();
    subtractionFilter->SetInput1( castFilter->GetOutput() );
    subtractionFilter->SetInput2( image );

    try
    {
      std::cout << "Subtracting registration images" << std::endl;
      subtractionFilter->Update();
    }
    catch (ExceptionObject &ex)
    {
      std::cerr << "ERROR: Could subtract images" << std::endl 
                << ex << std::endl;
      throw( ex );
    }

    // Rescale the output

    typedef typename itk::MinimumMaximumImageCalculator<OutputImageType> MinMaxCalculatorType;

    typename MinMaxCalculatorType::Pointer rangeCalculator = MinMaxCalculatorType::New();

    rangeCalculator->SetImage( subtractionFilter->GetOutput() );
    rangeCalculator->Compute();

    InputPixelType imMaximum = rangeCalculator->GetMaximum();
    InputPixelType imMinimum = rangeCalculator->GetMinimum();
    
    if ( imMaximum > -imMinimum )
    {
      imMinimum = -imMaximum;
    }
    else
    {
      imMaximum = -imMinimum;
    }

    typedef itk::Image<unsigned char, InputDimension>  ImageTypeUCHAR;

    typedef itk::IntensityWindowingImageFilter <OutputImageType, ImageTypeUCHAR> RescaleFilterType;
 
    typename RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
    rescaleFilter->SetInput( subtractionFilter->GetOutput() );

    rescaleFilter->SetWindowMinimum( imMinimum );
    rescaleFilter->SetWindowMaximum( imMaximum );

    rescaleFilter->SetOutputMinimum(         0 );
    rescaleFilter->SetOutputMaximum(       255 );


    typedef ImageFileWriter< ImageTypeUCHAR > FileWriterType;

    std::string fileModifiedOutput = BuildOutputFilename( fileInput, suffix );

    if ( niftk::FileExists( fileModifiedOutput ) ) 
    {
      std::cerr << std::endl << "WARNING: File " << fileModifiedOutput << " exists"
                << std::endl << "         but will be overwritten."
                << std::endl << std::endl;
    }

    typename FileWriterType::Pointer writer = FileWriterType::New();

    typename ImageIOBase::Pointer imageIO;
    imageIO = ImageIOFactory::CreateImageIO(fileModifiedOutput.c_str(), 
                                            ImageIOFactory::WriteMode);

    imageIO->SetMetaDataDictionary( dictionary );

    writer->SetFileName( fileModifiedOutput.c_str() );
    writer->SetInput( rescaleFilter->GetOutput() );
    writer->SetImageIO( imageIO );
    writer->UseInputMetaDataDictionaryOff();

    try
    {
      std::cout << "Writing " << description << " to file: "
                << fileModifiedOutput.c_str() << std::endl;
      writer->Update();
    }
  
    catch (ExceptionObject &ex)
    {
      std::cerr << "ERROR: Could notwrite file: " << fileModifiedOutput << std::endl 
                << ex << std::endl;
      throw( ex );
    }
  }
  else
  {
    itkExceptionMacro( << "Failed to write " << description 
                       << " to file - filename is empty " );
  }
};


// --------------------------------------------------------------------------
// RegisterTheImages()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::RegisterTheImages( void )
{

  m_RegistrationPreDiag = 
    RegisterTheImages( m_ImPreDiagnostic,
                       m_FilePreDiagnosticRegn,
                       m_ImPreDiagnosticMask,
                       
                       BuildOutputFilename( m_FileDiagnostic, 
                                            "_PreDiagReg2Diag_AffineTransform.txt" ),
                       BuildOutputFilename( m_FileDiagnostic, 
                                            "_PreDiagReg2Diag_AffineRegistered.nii.gz" ),
                       
                       BuildOutputFilename( m_FileDiagnostic, 
                                            "_PreDiagReg2Diag_NonRigidTransform.nii.gz" ),
                       BuildOutputFilename( m_FileDiagnostic, 
                                            "_PreDiagReg2Diag_NonRigidRegistered.nii.gz" ) );

  m_RegistrationControl = 
    RegisterTheImages( m_ImControl,
                       m_FileControlRegn,
                       m_ImControlMask,
                       
                       BuildOutputFilename( m_FileDiagnostic, 
                                            "_ControlReg2Diag_AffineTransform.txt" ),
                       BuildOutputFilename( m_FileDiagnostic, 
                                            "_ControlReg2Diag_AffineRegistered.nii.gz" ),
                       
                       BuildOutputFilename( m_FileDiagnostic, 
                                            "_ControlReg2Diag_NonRigidTransform.nii.gz" ),
                       BuildOutputFilename( m_FileDiagnostic, 
                                            "_ControlReg2Diag_NonRigidRegistered.nii.gz" ) );

};


// --------------------------------------------------------------------------
// RegisterTheImages()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
typename RegionalMammographicDensity< InputPixelType, InputDimension >::RegistrationFilterType::Pointer
RegionalMammographicDensity< InputPixelType, InputDimension >
::RegisterTheImages( typename ImageType::Pointer imSource,
                     std::string fileInputSource,
                     typename ImageType::Pointer maskSource,

                     std::string fileOutputAffineTransformation,
                     std::string fileOutputAffineRegistered,

                     std::string fileOutputNonRigidTransformation,
                     std::string fileOutputNonRigidRegistered )
{


  typename RegistrationFilterType::Pointer registration = RegistrationFilterType::New();

  if ( m_FlgVerbose )
  {
    registration->SetVerboseOn();
  }

  if ( m_FlgRegisterNonRigid )
  {
    registration->SetRegisterNonRigid();
  }

  registration->SetTypeOfInputImagesToRegister( m_TypeOfInputImagesToRegister );

  registration->SetWorkingDirectory( m_DirOutput );

  registration->SetTargetImage( m_ImDiagnostic );
  registration->SetSourceImage( imSource );

  registration->SetFileTarget( m_FileDiagnosticRegn );
  registration->SetFileSource( fileInputSource );

  registration->SetTargetMask( m_ImDiagnosticMask );
  registration->SetSourceMask( maskSource );

  registration->SetFileInputTargetRegistrationMask( m_FileDiagnosticRegnMask );

  registration->SetFileOutputAffineTransformation( fileOutputAffineTransformation );
  registration->SetFileOutputNonRigidTransformation( fileOutputNonRigidTransformation );

  registration->SetFileOutputAffineRegistered( fileOutputAffineRegistered );
  registration->SetFileOutputNonRigidRegistered( fileOutputNonRigidRegistered );


  std::cout << std::endl << "Starting the registration" << std::endl;
  registration->Update();
 
  typename ImageType::Pointer imAffineRegistered   = registration->GetOutput( 0 );
  typename ImageType::Pointer imNonRigidRegistered = registration->GetOutput( 1 );

  return registration;
};


#if 0

// --------------------------------------------------------------------------
// RegisterTheImages()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::RegisterTheImages( MammogramType mammoType,
                     typename FactoryType::EulerAffineTransformType::Pointer &transform )
{
  int finalInterpolator;
  int registrationInterpolator;
  int similarityMeasure;
  int transformation;
  int registrationStrategy;
  int optimizer;
  int bins;
  int iterations;
  int dilations;
  int levels;
  int startLevel;
  int stopLevel;
  double lowerIntensity;
  double higherIntensity;
  double dummyDefault;
  double paramTol;
  double funcTol;
  double maxStep;
  double minStep;
  double gradTol;
  double relaxFactor;
  double learningRate;
  double maskMinimumThreshold;
  double maskMaximumThreshold;
  double intensityFixedLowerBound;
  double intensityFixedUpperBound;
  double intensityMovingLowerBound;
  double intensityMovingUpperBound;
  double movingImagePadValue;
  int symmetricMetric;
  bool isRescaleIntensity;
  bool userSetPadValue;
  bool useWeighting; 
  double weightingThreshold; 
  double parameterChangeTolerance; 
  bool useCogInitialisation; 

  std::string outputMatrixTransformFile; 
  std::string outputUCLTransformFile;


  if ( mammoType == PREDIAGNOSTIC_MAMMO )
  {
    outputMatrixTransformFile = BuildOutputFilename( m_FileDiagnostic, "_PreDiagReg2Diag_Matrix.txt" );
    outputUCLTransformFile    = BuildOutputFilename( m_FileDiagnostic, "_PreDiagReg2Diag_UCLTransform.txt" );
  }
  else
  {
    outputMatrixTransformFile = BuildOutputFilename( m_FileDiagnostic, "_ControlReg2Diag_Matrix.txt" );
    outputUCLTransformFile    = BuildOutputFilename( m_FileDiagnostic, "_ControlReg2Diag_UCLTransform.txt" );
  }


  // Set defaults
  finalInterpolator = 4;
  registrationInterpolator = 2;
  similarityMeasure = 9;
  transformation = 4;
  registrationStrategy = 1;
  optimizer = 6;
  bins = 32;
  iterations = 300;
  dilations = 0;
  levels = 5;
  startLevel = 0;
  stopLevel = 2;
  lowerIntensity = 0;
  higherIntensity = 0;
  dummyDefault = -987654321;
  paramTol = 0.01;
  funcTol = 0.01;
  maxStep = 5.0;
  minStep = 0.01;
  gradTol = 0.01;
  relaxFactor = 0.5;
  learningRate = 0.5;
  maskMinimumThreshold = 0.5;
  maskMaximumThreshold = 255;
  intensityFixedLowerBound = dummyDefault;
  intensityFixedUpperBound = dummyDefault;
  intensityMovingLowerBound = dummyDefault;
  intensityMovingUpperBound = dummyDefault;
  movingImagePadValue = 0;
  symmetricMetric = 0;
  isRescaleIntensity = false;
  userSetPadValue = true;
  useWeighting = false; 
  useCogInitialisation = true; 
  
  // The factory.

  typename FactoryType::Pointer factory = FactoryType::New();
  
  // Start building.

  typename BuilderType::Pointer builder = BuilderType::New();

  builder->StartCreation( (itk::SingleResRegistrationMethodTypeEnum) registrationStrategy );
  builder->CreateInterpolator( (itk::InterpolationTypeEnum) registrationInterpolator );

  typename SimilarityMeasureType::Pointer metric = builder->CreateMetric( (itk::MetricTypeEnum) similarityMeasure );

  metric->SetSymmetricMetric( symmetricMetric );
  metric->SetUseWeighting( useWeighting ); 

  if (useWeighting)
  {
    metric->SetWeightingDistanceThreshold( weightingThreshold ); 
  }
  
  transform = dynamic_cast< typename FactoryType::EulerAffineTransformType* >( builder->CreateTransform((itk::TransformTypeEnum) transformation, 
                                                                                                          static_cast<const ImageType * >( m_ImDiagnostic ) ).GetPointer() );
  int dof = transform->GetNumberOfDOF(); 

  if ( niftk::FileExists( outputUCLTransformFile ) && ( ! m_FlgOverwrite ) )
  {
    std::cout << "Reading the registration transformation: " << outputUCLTransformFile << std::endl;
    transform = dynamic_cast<typename FactoryType::EulerAffineTransformType*>(builder->CreateTransform( outputUCLTransformFile ).GetPointer());
    transform->SetNumberOfDOF(dof); 
    if ( m_FlgDebug )
    {
      transform->Print( std::cout );
    }
    return;
  }
    
  // Compute and initial registration using the image moments

  InitialiseTransformationFromImageMoments( mammoType, transform );


  typename ImageMomentCalculatorType::VectorType fixedImgeCOG; 
  typename ImageMomentCalculatorType::VectorType movingImgeCOG; 

  fixedImgeCOG.Fill(0.); 
  movingImgeCOG.Fill(0.); 
  
  // Calculate the CoG for the initialisation using CoG or for the symmetric transformation. 

  if (useCogInitialisation || symmetricMetric == 2)
  {
    typename ImageMomentCalculatorType::Pointer fixedImageMomentCalculator = ImageMomentCalculatorType::New(); 

    fixedImageMomentCalculator->SetImage(m_ImDiagnostic); 
    fixedImageMomentCalculator->Compute(); 
    fixedImgeCOG = fixedImageMomentCalculator->GetCenterOfGravity(); 

    typename ImageMomentCalculatorType::Pointer movingImageMomentCalculator = ImageMomentCalculatorType::New(); 

    if ( mammoType == PREDIAGNOSTIC_MAMMO )
    {
      movingImageMomentCalculator->SetImage(m_ImPreDiagnostic); 
    }
    else
    {
      movingImageMomentCalculator->SetImage(m_ImControl); 
    }

    movingImageMomentCalculator->Compute(); 
    movingImgeCOG = movingImageMomentCalculator->GetCenterOfGravity(); 
  }
  
  if (symmetricMetric == 2)
  {
    builder->CreateFixedImageInterpolator( (itk::InterpolationTypeEnum) registrationInterpolator );
    builder->CreateMovingImageInterpolator( (itk::InterpolationTypeEnum) registrationInterpolator );
    
    // Change the center of the transformation for the symmetric transform. 

    typename ImageType::PointType centerPoint;

    for (unsigned int i = 0; i < InputDimension; i++)
      centerPoint[i] = (fixedImgeCOG[i] + movingImgeCOG[i])/2.; 

    typename FactoryType::EulerAffineTransformType::FullAffineTransformType* fullAffineTransform = transform->GetFullAffineTransform();

    int dof = transform->GetNumberOfDOF();
    transform->SetCenter(centerPoint);
    transform->SetNumberOfDOF(dof);
  }
  
  // Initialise the transformation using the CoG. 

  if (useCogInitialisation)
  {
    if (symmetricMetric == 2)
    {
      transform->InitialiseUsingCenterOfMass(fixedImgeCOG/2.0, 
                                               movingImgeCOG/2.0); 
    }
    else
    {
      transform->InitialiseUsingCenterOfMass(fixedImgeCOG, 
                                               movingImgeCOG); 

      typename ImageType::PointType centerPoint;

      centerPoint[0] = fixedImgeCOG[0];
      centerPoint[1] = fixedImgeCOG[1];

      transform->SetCenter(centerPoint);
    }
  }
  
  builder->CreateOptimizer((itk::OptimizerTypeEnum)optimizer);

  // Get the single res method.

  typename SingleResImageRegistrationMethodType::Pointer singleResMethod = builder->GetSingleResolutionImageRegistrationMethod();
  typename MultiResImageRegistrationMethodType::Pointer multiResMethod = MultiResImageRegistrationMethodType::New();

  if ( m_FlgDebug )
  {
    singleResMethod->SetDebug( true );
    multiResMethod->SetDebug( true );
  }

  // Sort out metric and optimizer  

  typedef typename itk::SimilarityMeasure< ImageType, ImageType >  SimilarityType;
  typedef SimilarityType* SimilarityPointer;

  SimilarityPointer similarityPointer = dynamic_cast< SimilarityPointer >(singleResMethod->GetMetric());
  
  if (optimizer == itk::SIMPLEX)
  {
    typedef typename itk::UCLSimplexOptimizer OptimizerType;
    typedef OptimizerType*                    OptimizerPointer;

    OptimizerPointer op = dynamic_cast< OptimizerPointer >( singleResMethod->GetOptimizer() );

    op->SetMaximumNumberOfIterations (iterations );
    op->SetParametersConvergenceTolerance( paramTol );
    op->SetFunctionConvergenceTolerance( funcTol );
    op->SetAutomaticInitialSimplex( true );
    op->SetMaximize(similarityPointer->ShouldBeMaximized());

    OptimizerType::ScalesType scales = transform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }
  else if (optimizer == itk::GRADIENT_DESCENT)
  {
    typedef typename itk::GradientDescentOptimizer OptimizerType;
    typedef OptimizerType*                         OptimizerPointer;
    OptimizerPointer op = dynamic_cast< OptimizerPointer >(singleResMethod->GetOptimizer());
    op->SetNumberOfIterations(iterations);
    op->SetLearningRate(learningRate);
    op->SetMaximize(similarityPointer->ShouldBeMaximized());

    OptimizerType::ScalesType scales = transform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }
  else if (optimizer == itk::REGSTEP_GRADIENT_DESCENT)
  {
    typedef typename itk::UCLRegularStepGradientDescentOptimizer OptimizerType;
    typedef OptimizerType*                                       OptimizerPointer;
    OptimizerPointer op = dynamic_cast< OptimizerPointer >(singleResMethod->GetOptimizer());
    op->SetNumberOfIterations(iterations);
    op->SetMaximumStepLength(maxStep);
    op->SetMinimumStepLength(minStep);
    op->SetRelaxationFactor(relaxFactor);
    op->SetMaximize(similarityPointer->ShouldBeMaximized());

    OptimizerType::ScalesType scales = transform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }
  else if (optimizer == itk::POWELL)
  {
    typedef typename itk::PowellOptimizer OptimizerType;
    typedef OptimizerType*                OptimizerPointer;
    OptimizerPointer op = dynamic_cast< OptimizerPointer >(singleResMethod->GetOptimizer());
    op->SetMaximumIteration(iterations);
    op->SetStepLength(maxStep);
    op->SetStepTolerance(minStep);
    op->SetMaximumLineIteration(10);
    op->SetValueTolerance(0.0001);
    op->SetMaximize(similarityPointer->ShouldBeMaximized());      

    OptimizerType::ScalesType scales = transform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }
  else if (optimizer == itk::SIMPLE_REGSTEP)
  {
    typedef typename itk::UCLRegularStepOptimizer OptimizerType;
    typedef OptimizerType*                        OptimizerPointer;
    OptimizerPointer op = dynamic_cast< OptimizerPointer >(singleResMethod->GetOptimizer());
    op->SetNumberOfIterations(iterations);
    op->SetMaximumStepLength(maxStep);
    op->SetMinimumStepLength(minStep);
    op->SetMaximize(similarityPointer->ShouldBeMaximized());
#if 1
    OptimizerType::ScalesType scales = transform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
#else
    OptimizerType::ScalesType scales;
    scales = SetRegistrationParameterScales< typename OptimizerType::ScalesType >((itk::TransformTypeEnum) transformation,
                                                                                  singleResMethod->GetTransform()->GetNumberOfParameters());
    op->SetScales(scales);      
#endif
  }
  else if (optimizer == itk::UCLPOWELL)
  {
    typedef itk::UCLPowellOptimizer OptimizerType;
    typedef OptimizerType*       OptimizerPointer;
    OptimizerPointer op = dynamic_cast< OptimizerPointer >(singleResMethod->GetOptimizer());
    op->SetMaximumIteration(iterations);
    op->SetStepLength(maxStep);
    op->SetStepTolerance(minStep);
    op->SetMaximumLineIteration(15);
    op->SetValueTolerance(1.0e-14);
    op->SetParameterTolerance(parameterChangeTolerance);
    op->SetMaximize(similarityPointer->ShouldBeMaximized());      

    OptimizerType::ScalesType scales = transform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }

  // Finish configuring single-res object
  singleResMethod->SetNumberOfDilations(dilations);
  singleResMethod->SetThresholdFixedMask(true);
  singleResMethod->SetThresholdMovingMask(true);  
  singleResMethod->SetFixedMaskMinimum(maskMinimumThreshold);
  singleResMethod->SetMovingMaskMinimum(maskMinimumThreshold);
  singleResMethod->SetFixedMaskMaximum(maskMaximumThreshold);
  singleResMethod->SetMovingMaskMaximum(maskMaximumThreshold);
  
  if (isRescaleIntensity)
  {
    singleResMethod->SetRescaleFixedImage(true);
    singleResMethod->SetRescaleFixedMinimum((InputPixelType)lowerIntensity);
    singleResMethod->SetRescaleFixedMaximum((InputPixelType)higherIntensity);
    singleResMethod->SetRescaleMovingImage(true);
    singleResMethod->SetRescaleMovingMinimum((InputPixelType)lowerIntensity);
    singleResMethod->SetRescaleMovingMaximum((InputPixelType)higherIntensity);
  }
  
  // Finish configuring multi-res object.
  multiResMethod->SetInitialTransformParameters( singleResMethod->GetTransform()->GetParameters() );
  multiResMethod->SetSingleResMethod(singleResMethod);
  if (stopLevel > levels - 1)
  {
    stopLevel = levels - 1;
  }  
  multiResMethod->SetNumberOfLevels(levels);
  multiResMethod->SetStartLevel(startLevel);
  multiResMethod->SetStopLevel(stopLevel);

  if (intensityFixedLowerBound != dummyDefault ||
      intensityFixedUpperBound != dummyDefault ||
      intensityMovingLowerBound != dummyDefault ||
      intensityMovingUpperBound != dummyDefault)
  {
    if (isRescaleIntensity)
    {
      singleResMethod->SetRescaleFixedImage(true);
      singleResMethod->SetRescaleFixedBoundaryValue(lowerIntensity);
      singleResMethod->SetRescaleFixedLowerThreshold(intensityFixedLowerBound);
      singleResMethod->SetRescaleFixedUpperThreshold(intensityFixedUpperBound);
      singleResMethod->SetRescaleFixedMinimum((InputPixelType)lowerIntensity+1);
      singleResMethod->SetRescaleFixedMaximum((InputPixelType)higherIntensity);
          
      singleResMethod->SetRescaleMovingImage(true);
      singleResMethod->SetRescaleMovingBoundaryValue(lowerIntensity);
      singleResMethod->SetRescaleMovingLowerThreshold(intensityMovingLowerBound);
      singleResMethod->SetRescaleMovingUpperThreshold(intensityMovingUpperBound);
      singleResMethod->SetRescaleMovingMinimum((InputPixelType)lowerIntensity+1);
      singleResMethod->SetRescaleMovingMaximum((InputPixelType)higherIntensity);

      metric->SetIntensityBounds(lowerIntensity+1, higherIntensity, lowerIntensity+1, higherIntensity);
    }
    else
    {
      metric->SetIntensityBounds(intensityFixedLowerBound, intensityFixedUpperBound, intensityMovingLowerBound, intensityMovingUpperBound);
    }
  }

  try
  {
    // The main filter.
    typename RegistrationFilterType::Pointer filter = RegistrationFilterType::New();
    filter->SetMultiResolutionRegistrationMethod(multiResMethod);

    std::cout << "Setting fixed image"<< std::endl;
    filter->SetFixedImage(m_ImDiagnostic);
    if ( m_FlgDebug ) m_ImDiagnostic->Print( std::cout );

    std::cout << "Setting moving image"<< std::endl;
    if ( mammoType == PREDIAGNOSTIC_MAMMO )
    {
      filter->SetMovingImage(m_ImPreDiagnostic);
      if ( m_FlgDebug ) m_ImPreDiagnostic->Print( std::cout );
    }
    else
    {
      filter->SetMovingImage(m_ImControl);
      if ( m_FlgDebug ) m_ImControl->Print( std::cout );
    }

    std::cout << "Setting fixed mask"<< std::endl;
    filter->SetFixedMask(m_ImDiagnosticRegnMask);  
    if ( m_FlgDebug ) m_ImDiagnosticRegnMask->Print( std::cout );

    std::cout << "Setting moving mask"<< std::endl;
    if ( mammoType == PREDIAGNOSTIC_MAMMO )
    {
      filter->SetMovingMask(m_ImPreDiagnosticRegnMask);
      if ( m_FlgDebug ) m_ImPreDiagnosticRegnMask->Print( std::cout );
    }
    else
    {
      filter->SetMovingMask(m_ImControlRegnMask);
      if ( m_FlgDebug ) m_ImControlRegnMask->Print( std::cout );
    }

    // If we havent asked for output, turn off reslicing.
    filter->SetDoReslicing(true);

    filter->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)finalInterpolator));
    
    // Set the padding value
    if (!userSetPadValue)
    {
      typename ImageType::IndexType index;
      for (unsigned int i = 0; i < InputDimension; i++)
      {
        index[i] = 0;  
      }
      if ( mammoType == PREDIAGNOSTIC_MAMMO )
      {
        movingImagePadValue = m_ImPreDiagnostic->GetPixel(index);
      }
      else
      {
        movingImagePadValue = m_ImControl->GetPixel(index);
      }
      std::cout << "Setting  moving image pad value to:" 
        + niftk::ConvertToString(movingImagePadValue)<< std::endl;
    }
    similarityPointer->SetTransformedMovingImagePadValue(movingImagePadValue);
    filter->SetResampledMovingImagePadValue(movingImagePadValue);
    
    // Run the registration
    filter->Update();
    
    // And write the output.
    if ( mammoType == PREDIAGNOSTIC_MAMMO )
    {
      WriteImageFile< OutputImageType >( m_FilePreDiagnostic, 
                                         std::string( "_PreDiagReg2Diag.dcm" ), 
                                         "registered pre-diagnostic image", 
                                         filter->GetOutput(), 
                                         m_DiagDictionary );

      WriteRegistrationDifferenceImage( m_FilePreDiagnostic, 
                                        std::string( "_PreDiagReg2DiagDifference.jpg" ), 
                                        "registered pre-diagnostic difference image", 
                                        filter->GetOutput(), 
                                        m_DiagDictionary );
    }
    else
    {
      WriteImageFile< OutputImageType >( m_FileControl, 
                                         std::string( "_ControlReg2Diag.dcm" ), 
                                         "registered control image", 
                                         filter->GetOutput(), 
                                         m_DiagDictionary );

      WriteRegistrationDifferenceImage( m_FileControl, 
                                        std::string( "_ControlReg2DiagDifference.jpg" ), 
                                        "registered control difference image", 
                                        filter->GetOutput(), 
                                        m_DiagDictionary );
    }

    // Make sure we get the final one.
    transform = dynamic_cast< typename FactoryType::EulerAffineTransformType* >(singleResMethod->GetTransform());
    transform->SetFullAffine(); 
    
    // Save the transform (as 12 parameter UCLEulerAffine transform).
    typedef typename itk::TransformFileWriter TransformFileWriterType;
    typename TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();
    transformFileWriter->SetInput(transform);
    transformFileWriter->SetFileName(outputUCLTransformFile);
    transformFileWriter->Update();         
    
    // Save the transform (as 16 parameter matrix transform).
    if (outputMatrixTransformFile.length() > 0)
    {
      transformFileWriter->SetInput( transform->GetFullAffineTransform() );
      transformFileWriter->SetFileName( outputMatrixTransformFile );
      transformFileWriter->Update(); 
    }

    if ( mammoType == PREDIAGNOSTIC_MAMMO )
    {
      m_TransformPreDiag = transform;
    }
    else
    {
      m_TransformControl = transform;
    }
    
  }
  catch( itk::ExceptionObject & excp )
  {
    throw( excp );
  }
};

#endif


// --------------------------------------------------------------------------
// TransformTumourPositionIntoImage()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
typename RegionalMammographicDensity< InputPixelType, InputDimension >::LabelImageType::IndexType
RegionalMammographicDensity< InputPixelType, InputDimension >
::TransformTumourPositionIntoImage( typename LabelImageType::IndexType &idxTumourCenter,
                                    typename ImageType::Pointer &image,
                                    typename RegistrationFilterType::Pointer registration )
{
  if ( ! registration )
  {
    itkExceptionMacro( << "ERROR: Cannot transform tumour position - no registration available" );
  }

  typename LabelImageType::IndexType centerIndex;
  typename LabelImageType::PointType inPoint;
  typename LabelImageType::PointType outPoint;


  m_ImDiagnostic->TransformIndexToPhysicalPoint( idxTumourCenter, inPoint );

  outPoint = registration->TransformPoint( inPoint );

  image->TransformPhysicalPointToIndex( outPoint, centerIndex );

  if ( m_FlgDebug )
  {
    std::cout << "Tumour center: " << idxTumourCenter 
              << ", point: " << inPoint << std::endl
              << "  transforms to point: " << outPoint
              << ", index: " << centerIndex << std::endl
              << std::endl;
  }

  return centerIndex;
};


// --------------------------------------------------------------------------
// GenerateRandomTumourPositionInImage()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::GenerateRandomTumourPositionInImage( MammogramType mammoType )
{
  // Create a distance transform of the pre-diagnostic mask

  typename DistanceTransformType::Pointer distanceTransform = DistanceTransformType::New();

  if ( mammoType == PREDIAGNOSTIC_MAMMO )
  {
   distanceTransform->SetInput( m_ImPreDiagnosticMask );   
  }
  else
  {
    distanceTransform->SetInput( m_ImControlMask );
  }

  distanceTransform->SetInsideIsPositive( true );
  distanceTransform->UseImageSpacingOn();
  distanceTransform->SquaredDistanceOff();

  try
  {
    std::cout << "Computing distance transform for mask" << std::endl;
    distanceTransform->UpdateLargestPossibleRegion();
  }    
  catch (ExceptionObject &ex)
  {
    std::cerr << ex << std::endl;
    throw( ex );
  }

  if ( m_FlgDebug )
  {
    if ( mammoType == PREDIAGNOSTIC_MAMMO )
    {
      WriteImageFile< RealImageType >( m_FilePreDiagnostic, 
                                       std::string( "_PreDiagMaskDistance.nii.gz" ), 
                                       "mask distance transform for pre-diagnostic image", 
                                       distanceTransform->GetOutput(), 
                                       m_PreDiagDictionary );
    }
    else
    {
      WriteImageFile< RealImageType >( m_FileControl, 
                                       std::string( "_ControlMaskDistance.nii.gz" ), 
                                       "mask distance transform for control image", 
                                       distanceTransform->GetOutput(), 
                                       m_ControlDictionary );
    }
  }

  // Threshold the distance transform to generate a mask for the internal pre-diag patches
  
  typedef itk::Image< unsigned char, InputDimension > MaskImageType;
  typedef typename itk::BinaryThresholdImageFilter<RealImageType, MaskImageType> BinaryThresholdFilterType;
  

  typename BinaryThresholdFilterType::Pointer thresholdFilter = BinaryThresholdFilterType::New();

  RealType threshold = sqrt( 2.*m_RegionSizeInMM*m_RegionSizeInMM )/2.;

  thresholdFilter->SetInput( distanceTransform->GetOutput() );

  thresholdFilter->SetOutsideValue( 0 );
  thresholdFilter->SetInsideValue( 255 );
  thresholdFilter->SetLowerThreshold( threshold );

  try
  {
    std::cout << "Thresholding distance transform of pre-diagnostic mask at: "
              << threshold << std::endl;
    thresholdFilter->UpdateLargestPossibleRegion();
  }    
  catch (ExceptionObject &ex)
  {
    std::cerr << ex << std::endl;
    throw( ex );
  }

  typename MaskImageType::Pointer mask = thresholdFilter->GetOutput();
  mask->DisconnectPipeline();

  if ( m_FlgDebug )
  {
    if ( mammoType == PREDIAGNOSTIC_MAMMO )
    {
      WriteImageFile< MaskImageType >( m_FilePreDiagnostic, 
                                       std::string( "_PreDiagPatchMask.nii.gz" ), 
                                       "patch mask for pre-diagnostic image", 
                                       mask, 
                                       m_PreDiagDictionary );
    }
    else
    {
      WriteImageFile< MaskImageType >( m_FileControl, 
                                       std::string( "_ControlPatchMask.nii.gz" ), 
                                       "patch mask for control image", 
                                       mask, 
                                       m_ControlDictionary );
    } 
  }

  // Select a random pixel inside this mask to use as the pre-diagnostic tumour position

  bool found = false;
  typename MaskImageType::SizeType  size;
  typename MaskImageType::IndexType idx;

  size = mask->GetLargestPossibleRegion().GetSize();

  if ( m_FlgDebug )
  {
    std::cout << "Random pixel distributions, x: 0 to " << size[0] 
              << " y: 0 to " << size[1] << std::endl;
  }

  boost::random::uniform_int_distribution<> xdist(0, size[0] - 1);
  boost::random::uniform_int_distribution<> ydist(0, size[1] - 1);

  while ( ! found ) 
  {
    idx[0] =  xdist( m_Gen );
    idx[1] =  ydist( m_Gen );

    if ( m_FlgDebug )
    {
      std::cout << "Random pixel coordinate: (" << idx[0] << ", " << idx[1] << ")" << std::endl;
    }
    
    if ( mask->GetPixel( idx ) )
    {
      found = true;
    }
  }
  
  if ( mammoType == PREDIAGNOSTIC_MAMMO )
  {
    m_PreDiagCenterIndex = idx;

    std::cout << "Random region in pre-diagnostic image will be centered on pixel: "
              << m_PreDiagCenterIndex << std::endl;
  }
  else
  {
    m_ControlCenterIndex = idx;

    std::cout << "Random region in control image will be centered on pixel: "
              << m_ControlCenterIndex << std::endl;
  }

>>>>>>> Issue #4270: Modification of RegionalMammographicDensity to use MammogramRegistrationFilter
};


} // namespace itk
