/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#include <itkRegionalMammographicDensity.h>

namespace itk
{


// --------------------------------------------------------------------------
// Constructor
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
RegionalMammographicDensity< InputPixelType, InputDimension >
::RegionalMammographicDensity()
{
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

  if ( this->m_ImDiagnostic && this->m_ImPreDiagnostic )
  {
    std::cout << "Registering diagnostic and pre-diagnostic images" << std::endl;

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
  else
  {
    std::cout << "Could not register diagnostic and pre-diagnostic images"
              << ", images not set."<< std::endl;
  }

  // The control image registration

  if ( this->m_ImDiagnostic && this->m_ImControl )
  {
    std::cout << "Registering diagnostic and control images" << std::endl;

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
  else
  {
    std::cout << "Could not register diagnostic and control images"
              << ", images not set."<< std::endl;
  }

};


// --------------------------------------------------------------------------
// Compute()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
bool
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
        return false;
      }

      this->m_ImDiagnosticMask = reader->GetOutput();
      this->m_ImDiagnosticMask->DisconnectPipeline();

      // And registration version?

      if ( this->m_FlgRegister || this->m_FlgRegisterNonRigid )
      {
        this->m_FileDiagnosticRegnMask = this->BuildOutputFilename( this->m_FileDiagnostic, diagRegnMaskSuffix );

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
            return false;
          }

          this->m_ImDiagnosticRegnMask = reader->GetOutput();
          this->m_ImDiagnosticRegnMask->DisconnectPipeline();
        }
        else
        {
          itkExceptionMacro( << "ERROR: Cannot read diagnostic registration mask: " << this->m_FileDiagnosticRegnMask );
          return false;
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
        return false;
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
            return false;
          }

          this->m_ImPreDiagnosticRegnMask = reader->GetOutput();
          this->m_ImPreDiagnosticRegnMask->DisconnectPipeline();
        }
        else
        {
          itkExceptionMacro( << "ERROR: Cannot read pre-diagnostic registration mask: " << fileRegnMask );
          return false;
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

  // Generate the Control Mask

  if ( this->m_ImControl )
  {

    fileMask = this->BuildOutputFilename( this->m_FileControl, controlMaskSuffix );

    if ( niftk::FileExists( fileMask ) && ( ! this->m_FlgOverwrite ) )
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
        return false;
      }

      this->m_ImControlMask = reader->GetOutput();
      this->m_ImControlMask->DisconnectPipeline();

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
            return false;
          }

          this->m_ImControlRegnMask = reader->GetOutput();
          this->m_ImControlRegnMask->DisconnectPipeline();
        }
        else
        {
          itkExceptionMacro( << "ERROR: Cannot read control registration mask: " << fileRegnMask );
          return false;
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
                                                     std::string( "_DiagLabels.nii.gz" ),
                                                     "diagnostic labels",
                                                     this->m_ImDiagnosticLabels,
                                                     this->m_DiagDictionary );
    }

    this->WriteLabelImageFile( this->m_FileDiagnostic,
                               std::string( "_DiagLabels.jpg" ),
                               "diagnostic labels",
                               this->m_ImDiagnosticLabels,
                               this->m_DiagTumourRegion,
                               this->m_DiagDictionary );
  }

  // Calculate the pre-diagnostic labels

  if ( this->m_ImPreDiagnostic )
  {
    if ( this->m_FlgVerbose )
      std::cout << "Computing pre-diagnostic mammo labels." << std::endl;

    if ( this->m_FlgRegister || this->m_FlgRegisterNonRigid )
    {
      this->m_PreDiagCenterIndex = this->TransformTumourPositionIntoImage( this->m_DiagTumourCenterIndex,
                                                                           this->m_ImPreDiagnostic,
                                                                           this->m_RegistrationPreDiag );

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
                                                     std::string( "_PreDiagLabels.nii.gz" ),
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

    if ( this->m_FlgDebug )
    {
      this->template WriteImageFile<LabelImageType>( this->m_FileControl,
                                                     std::string( "_ControlLabels.nii.gz" ),
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

  return true;
};


} // namespace itk
