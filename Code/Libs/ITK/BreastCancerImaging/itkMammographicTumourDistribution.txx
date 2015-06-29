/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#include <itkMammographicTumourDistribution.h>


namespace itk
{


// --------------------------------------------------------------------------
// Constructor
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
MammographicTumourDistribution< InputPixelType, InputDimension >
::MammographicTumourDistribution()
{
  m_foutOutputCSV = 0;

  this->SetRegisterOn();
};


// --------------------------------------------------------------------------
// Destructor
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
MammographicTumourDistribution< InputPixelType, InputDimension >
::~MammographicTumourDistribution()
{
}


// --------------------------------------------------------------------------
// RunRegistration()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
MammographicTumourDistribution< InputPixelType, InputDimension >
::RunRegistration( void )
{
  typename ImageType::Pointer imAffineRegistered;
  typename ImageType::Pointer imNonRigidRegistered;


  if ( this->m_ImDiagnostic && this->m_ImControl )
  {
    std::string fileOutput = niftk::ConcatenatePath( this->m_DirOutput,
						     this->m_FileDiagnostic );
    std::string dirOutput = fs::path( fileOutput ).branch_path().string();

    this->WriteRegistrationDifferenceImage( this->m_FileDiagnostic,
					    std::string( "_UnregisteredDifference.jpg" ),
					    "un-registered difference image",
					    this->m_ImControl,
					    this->m_DiagDictionary );

    this->m_RegistrationControl =
      this->RegisterTheImages( this->m_ImControl,
			       this->m_FileControlRegn,
			       this->m_ImControlMask,

			       this->BuildOutputFilename( this->m_FileDiagnostic,
							  "_AffineTransform.txt" ),
			       this->BuildOutputFilename( this->m_FileDiagnostic,
							  "_AffineRegistered.nii.gz" ),

			       this->BuildOutputFilename( this->m_FileDiagnostic,
							  "_NonRigidTransform.nii.gz" ),
			       this->BuildOutputFilename( this->m_FileDiagnostic,
							  "_NonRigidRegistered.nii.gz" ),
			       &dirOutput );

    imAffineRegistered   = this->m_RegistrationControl->GetOutput( 0 );

    this->WriteRegistrationDifferenceImage( this->m_FileDiagnostic,
					    std::string( "_AffineDifference.jpg" ),
					    "affine registered difference image",
					    imAffineRegistered,
					    this->m_DiagDictionary );

    imNonRigidRegistered = this->m_RegistrationControl->GetOutput( 1 );

    this->WriteRegistrationDifferenceImage( this->m_FileDiagnostic,
					    std::string( "_NonRigidDifference.jpg" ),
					    "non-rigidly registered difference image",
					    imNonRigidRegistered,
					    this->m_DiagDictionary );
  }

};


// --------------------------------------------------------------------------
// WriteDataToCSVFile()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
MammographicTumourDistribution< InputPixelType, InputDimension >
::WriteDataToCSVFile( std::ofstream *foutCSV )
{
  *foutCSV << std::setprecision( 9 )
	   << std::right << std::setw(10) << this->m_SetNumberDiagnostic << ", "
	   << std::right << std::setw(17) << this->m_IdDiagnosticImage << ", "
	   << std::right << std::setw(60) << this->m_FileDiagnostic << ", "

	   << std::right << std::setw(17) << this->m_SetNumberControl << ", "
	   << std::right << std::setw(17) << this->m_IdControlImage << ", "
	   << std::right << std::setw(60) << this->m_FileControl << ", "

	   << std::right << std::setw( 9) << this->m_StrTumourID << ", "
	   << std::right << std::setw(17) << this->m_StrTumourImageID << ", "

	   << std::right << std::setw(17) << this->m_DiagTumourCenterIndex[0] << ", "
	   << std::right << std::setw(17) << this->m_DiagTumourCenterIndex[1] << ", "

	   << std::right << std::setw(17) << this->m_ControlCenterIndex[0] << ", "
	   << std::right << std::setw(17) << this->m_ControlCenterIndex[1]

	   << std::endl;
};


// --------------------------------------------------------------------------
// Compute()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
bool
MammographicTumourDistribution< InputPixelType, InputDimension >
::Compute()
{
  std::string fileMask;
  std::string fileRegnMask;

  std::string diagMaskSuffix(       "_DiagMask.nii.gz" );
  std::string controlMaskSuffix( "_ControlMask.nii.gz" );

  std::string diagRegnMaskSuffix(       "_DiagRegnMask.nii.gz" );
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

	    this->m_ImDiagnosticRegnMask = reader->GetOutput();
	    this->m_ImDiagnosticRegnMask->DisconnectPipeline();
          }

          catch (ExceptionObject &ex)
          {
            std::cerr << "ERROR: Could not read file: "
                      << this->m_FileDiagnosticRegnMask << std::endl << ex << std::endl;
            throw( ex );
          }
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
        std::cout << "Reading the reference mask: " << fileMask << std::endl;

        reader->Update();

	this->m_ImControlMask = reader->GetOutput();
	this->m_ImControlMask->DisconnectPipeline();
      }

      catch (ExceptionObject &ex)
      {
        std::cerr << "ERROR: Could not read file: "
                  << fileMask << std::endl << ex << std::endl;
        throw( ex );
      }

      // And registration version?

      if ( this->m_FlgRegister || this->m_FlgRegisterNonRigid )
      {
        fileRegnMask = this->BuildOutputFilename( this->m_FileControl,
						  controlRegnMaskSuffix );

        if ( niftk::FileExists( fileRegnMask ) )
        {
          reader->SetFileName( fileRegnMask );

          try
          {
            std::cout << "Reading the reference registration mask: "
		      << fileRegnMask << std::endl;

            reader->Update();

	    this->m_ImControlRegnMask = reader->GetOutput();
	    this->m_ImControlRegnMask->DisconnectPipeline();
          }

          catch (ExceptionObject &ex)
          {
            std::cerr << "ERROR: Could not read file: "
                      << fileRegnMask << std::endl << ex << std::endl;
            throw( ex );
          }
        }
        else
        {
          itkExceptionMacro( << "ERROR: Cannot read reference registration mask: "
			     << fileRegnMask );
        }
      }
    }
    else
    {
      this->m_ImControlMask = this->MaskWithPolygon( Superclass::CONTROL_MAMMO );

      this->template CastImageAndWriteToFile< unsigned char >( this->m_FileControl,
							       controlMaskSuffix,
							       "reference mask",
							       this->m_ImControlMask,
							       this->m_ControlDictionary );
    }
  }

  // Register the images

  if ( this->m_ImDiagnostic && this->m_ImControl )
  {
    RunRegistration();
  }
  else
  {
    std::cerr << "WARNING: Diagnostic and control images are not both set, " << std::endl
              << "         aborting processing for this patient."
              << std::endl;
    return false;
  }

  // Save the diagnostic image with the tumour region inverted

  typename ImageType::Pointer imTumour = this->DrawTumourRegion( this->m_ImDiagnostic );

  if ( imTumour )
  {
    this->template CastImageAndWriteToFile< unsigned char >( this->m_FileDiagnostic,
                                                             std::string( "_Tumour.jpg" ),
                                                             "diagnostic tumour image",
                                                             imTumour,
                                                             this->m_DiagDictionary );
  }
  else
  {
    std::cerr << "WARNING: No diagnostic tumour region, aborting processing for this patient"
              << std::endl;
    return false;
  }


  // Compute the tumour location in the control image

  this->m_ControlCenterIndex = this->TransformTumourPositionIntoImage( this->m_DiagTumourCenterIndex,
								       this->m_ImControl,
								       this->m_RegistrationControl );

  if ( this->m_FlgVerbose )
    std::cout << "   Tumour center in reference image: "
	      << this->m_ControlCenterIndex[0] << ", "
	      << this->m_ControlCenterIndex[1] << std::endl;

  // Save the control image with the transformed tumour region inverted

  imTumour = this->DrawTumourOnReferenceImage();

  std::string dirOutput =
    fs::path( niftk::ConcatenatePath( this->m_DirOutput,
				      this->m_FileDiagnostic ) ).branch_path().string();

  std::string fileOutput = niftk::ConcatenatePath( dirOutput,
						   fs::path( this->m_FileControl ).filename().string() );

  fileOutput = niftk::ModifyImageFileSuffix( fileOutput,
					     std::string( "_Tumour.jpg" ) );

  this->template CastImageAndWriteToFile< unsigned char >( fileOutput,
							   "reference tumour image",
							   imTumour,
							   this->m_DiagDictionary );


  // Write the data to the output csv file

  std::string fileDiagnosticCSV = this->BuildOutputFilename( this->m_FileDiagnostic,
							     ".csv" );

  std::ofstream foutDiagnosticCSV( fileDiagnosticCSV.c_str(), std::ios::binary );

  if ( ( ! foutDiagnosticCSV ) ||
       foutDiagnosticCSV.bad() ||
       foutDiagnosticCSV.fail() )
  {
    std::cerr << "ERROR: Could not open CSV output file: "
	      << fileDiagnosticCSV << std::endl;
    return false;
  }
  else
  {
    WriteDataToCSVFile( &foutDiagnosticCSV );
    foutDiagnosticCSV.close();
  }

  WriteDataToCSVFile( m_foutOutputCSV );

  return true;
};


// --------------------------------------------------------------------------
// AddPointToTumourPolygon()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
MammographicTumourDistribution< InputPixelType, InputDimension >
::AddPointToTumourPolygon( typename PolygonType::Pointer &polygon, int i, int j)
{
  typename ImageType::IndexType index;

  typename ImageType::PointType inPoint;
  typename ImageType::PointType outPoint;

  index[0] = i;
  index[1] = j;

  this->m_ImDiagnostic->TransformIndexToPhysicalPoint( index, inPoint );

  outPoint = this->m_RegistrationControl->TransformPoint( inPoint );

  polygon->AddPoint( outPoint );
};


// --------------------------------------------------------------------------
// DrawTumourOnReferenceImage()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
typename MammographicTumourDistribution< InputPixelType, InputDimension >::ImageType::Pointer
MammographicTumourDistribution< InputPixelType, InputDimension >
::DrawTumourOnReferenceImage()
{
  typename ImageType::RegionType region;
  typename ImageType::Pointer imMask;

  // Create the polygon

  typename PolygonType::Pointer polygon = PolygonType::New();

  polygon->ComputeObjectToWorldTransform();

  // Add the points

  AddPointToTumourPolygon( polygon, this->m_TumourLeft,  this->m_TumourTop );
  AddPointToTumourPolygon( polygon, this->m_TumourLeft,  this->m_TumourBottom );
  AddPointToTumourPolygon( polygon, this->m_TumourRight, this->m_TumourBottom );
  AddPointToTumourPolygon( polygon, this->m_TumourRight, this->m_TumourTop );
  AddPointToTumourPolygon( polygon, this->m_TumourLeft,  this->m_TumourTop );

  // Create the mask

  typedef SpatialObjectToImageFilter< PolygonType, ImageType > SpatialObjectToImageFilterType;

  typename SpatialObjectToImageFilterType::Pointer
    polyMaskFilter = SpatialObjectToImageFilterType::New();

  region = this->m_ImControl->GetLargestPossibleRegion();

  polyMaskFilter->SetInput( polygon );
  polyMaskFilter->SetInsideValue( 1000 );
  polyMaskFilter->SetOutsideValue( 0 );

  polyMaskFilter->SetSize( region.GetSize() );
  polyMaskFilter->SetSpacing( this->m_ImControl->GetSpacing() );

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

  imMask->SetSpacing( this->m_ImControl->GetSpacing() );


  // If this is a right mammogram then flip it

  if ( this->m_BreastSideControl ==
       LeftOrRightSideCalculatorType::RIGHT_BREAST_SIDE )
  {
    itk::FixedArray<bool, 2> flipAxes;
    flipAxes[0] = true;
    flipAxes[1] = false;

    if ( this->m_FlgDebug )
    {
      std::cout << "This is a right mammogram so flipping the mask in 'x'"
		<< std::endl;
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

  // Use the mask to draw the tumour region on the reference image

  typename ImageType::Pointer image;

  typedef itk::ImageDuplicator< ImageType > DuplicatorType;

  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();

  duplicator->SetInputImage( this->m_ImControl );
  duplicator->Update();

  image = duplicator->GetOutput();
  image->DisconnectPipeline();


  typedef typename itk::MinimumMaximumImageCalculator<ImageType> MinMaxCalculatorType;

  typename MinMaxCalculatorType::Pointer rangeCalculator = MinMaxCalculatorType::New();

  rangeCalculator->SetImage( image );
  rangeCalculator->Compute();

  InputPixelType imMaximum = rangeCalculator->GetMaximum();
  InputPixelType imMinimum = rangeCalculator->GetMinimum();

  itk::ImageRegionIterator< ImageType > itImage( image, region );
  itk::ImageRegionIterator< ImageType > itMask( imMask, region );

  for ( itImage.GoToBegin(), itMask.GoToBegin();
	! itImage.IsAtEnd();
	++itImage, ++itMask )
  {
    if ( itMask.Get() )
    {
      itImage.Set( imMaximum - ( itImage.Get() - imMinimum ) );
    }
  }

  return image;
};


} // namespace itk
