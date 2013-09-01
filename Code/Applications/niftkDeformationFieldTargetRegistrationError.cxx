/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <niftkConversionUtils.h>
#include <niftkCommandLineParser.h>

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkAddImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkMultiplyImageFilter.h>

#include <itkImage.h>
#include <itkImageRegionIterator.h>

#include <itkEuler3DTransform.h>
#include <itkVectorResampleImageFilter.h>


struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_SWITCH, "dbg", 0, "Output debugging information."},
  {OPT_SWITCH, "v", 0,   "Verbose output during execution."},

  {OPT_INT, "mvalue", "value", "The mask intensity used to determine the region of interest."},
  {OPT_STRING, "mask", "filename", "Calculate error only over mask region."},

  {OPT_STRING, "oi", "filename", "Output an image of the error at each voxel."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "Input deformation field 1."},
  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "Input deformation field 2."},

  {OPT_DONE, NULL, NULL, 
   "Calculates the error between two deformation fields which act in the same direction. "
   "Error calculated at voxel positions of mask or deformation field 1."
  }
};


enum {
  O_DEBUG = 0,
  O_VERBOSE,

  O_MASK_VALUE,
  O_MASK,

  O_OUTPUT_IMAGE,

  O_DEFORMATION_1,
  O_DEFORMATION_2
};

  

int main( int argc, char ** argv )
{
  // Default filenames
  char *in1Name = NULL;
  char *in2Name = NULL;
  char *maskName = NULL;
  char *outName = NULL;

  const unsigned int Dimension = 3;


  typedef int MaskPixelType;
  MaskPixelType mask_value = 1;
  
  // Image and reader definitions
  
  typedef   float InternalPixelType;
  typedef   itk::Vector< InternalPixelType, Dimension > VectorPixelType;
  typedef   itk::Image< VectorPixelType,  Dimension >   FieldType;

  typedef FieldType::Pointer              FieldPointer;
  typedef itk::ImageRegionIterator<FieldType>  FieldIterator;
  typedef FieldType::PixelType            DisplacementType;
  
  typedef FieldType::IndexType     FieldIndexType;
  typedef FieldType::RegionType    FieldRegionType;
  typedef FieldType::SizeType      FieldSizeType;
  typedef FieldType::SpacingType   FieldSpacingType;
  typedef FieldType::PointType     FieldPointType;
  typedef FieldType::DirectionType FieldDirectionType;
  
  typedef itk::ImageFileReader < FieldType >  FieldReaderType;

  typedef itk::Image< MaskPixelType,  Dimension >    MaskImageType;
  typedef itk::ImageRegionIterator<MaskImageType>    MaskIterator;

  typedef itk::ImageFileReader< MaskImageType  >  MaskReaderType;

  typedef itk::Image< InternalPixelType,  Dimension >    InternalImageType;
  
  typedef itk::ImageFileWriter< InternalImageType  >  InternalImageWriterType;

  typedef itk::ImageRegionIterator<InternalImageType>  ImageIterator;


  // Declare the type for filters
  typedef itk::AddImageFilter<
  InternalImageType,
    InternalImageType,
    InternalImageType  >       addFilterType;
  
  typedef itk::SubtractImageFilter<
  InternalImageType,
    InternalImageType,
    InternalImageType  >       subFilterType;
  
  typedef itk::MultiplyImageFilter<
  InternalImageType,
    InternalImageType,
    InternalImageType > multiplyFilterType;


  typedef addFilterType::Pointer                addFilterTypePointer;
  typedef subFilterType::Pointer                subFilterTypePointer;
  typedef multiplyFilterType::Pointer           multiplyFilterTypePointer;

  // Parse the command line
  // ~~~~~~~~~~~~~~~~~~~~~~
  
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_MASK_VALUE, mask_value);
  CommandLineOptions.GetArgument(O_MASK, maskName);

  CommandLineOptions.GetArgument(O_OUTPUT_IMAGE, outName);

  CommandLineOptions.GetArgument(O_DEFORMATION_1, in1Name);
  CommandLineOptions.GetArgument(O_DEFORMATION_2, in2Name);


  // Create the image readers etc.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  FieldReaderType::Pointer fieldReader1 = FieldReaderType::New();
  FieldReaderType::Pointer fieldReader2 = FieldReaderType::New();
                                           
  FieldType::Pointer singleField1;
  FieldType::Pointer singleField2;

  MaskReaderType::Pointer maskReader = MaskReaderType::New();

  MaskImageType::Pointer mask;

  InternalImageWriterType::Pointer writer = InternalImageWriterType::New();

  InternalImageType::Pointer errorImage;
  InternalImageType::Pointer initialerrorImage;
    
  addFilterTypePointer addfilter = addFilterType::New();
  subFilterTypePointer subfilter = subFilterType::New();
  multiplyFilterTypePointer multiplyfilter = multiplyFilterType::New();


  // Prepare resampling of deformation field 2 if needed
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::VectorResampleImageFilter< FieldType, FieldType > FieldResampleFilterType;

  FieldResampleFilterType::Pointer fieldResample = FieldResampleFilterType::New();

  VectorPixelType zeroDisplacement;
  zeroDisplacement[0] = 0.0;
  zeroDisplacement[1] = 0.0;
  zeroDisplacement[2] = 0.0;

  typedef itk::Euler3DTransform< double > RigidTransformType;
  RigidTransformType::Pointer rigidIdentityTransform = RigidTransformType::New();
  rigidIdentityTransform->SetIdentity();


  // Read the deformation fields
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  fieldReader1->SetFileName(in1Name);
  try
    {
      fieldReader1->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
      std::cerr << "Exception thrown " << std::endl;
      std::cerr << excp << std::endl;
    }
  singleField1 = fieldReader1->GetOutput();
  //singleField1->Print(std::cout); 
  
  fieldReader2->SetFileName(in2Name);
  try
    {
      fieldReader2->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
      std::cerr << "Exception thrown " << std::endl;
      std::cerr << excp << std::endl;
    }
  singleField2 = fieldReader2->GetOutput();
  //singleField2->Print(std::cout); 

  FieldRegionType region = singleField1->GetLargestPossibleRegion();
  FieldSizeType size = singleField1->GetLargestPossibleRegion().GetSize();
  FieldPointType origin = singleField1->GetOrigin();
  FieldSpacingType spacing = singleField1->GetSpacing();
  FieldDirectionType direction = singleField1->GetDirection();
  

  // Read the mask image
  // ~~~~~~~~~~~~~~~~~~~

  if (maskName != NULL)
    {
      maskReader->SetFileName(maskName);
      try
	{
	  maskReader->Update();
	}
      catch( itk::ExceptionObject & excp )
	{
	  std::cerr << "Exception thrown " << std::endl;
	  std::cerr << excp << std::endl;
	}
      mask = maskReader->GetOutput();

      FieldRegionType regionM = mask->GetLargestPossibleRegion();
      FieldSizeType sizeM = mask->GetLargestPossibleRegion().GetSize();
      FieldPointType originM = mask->GetOrigin();
      FieldSpacingType spacingM = mask->GetSpacing();
      FieldDirectionType directionM = mask->GetDirection();
      

      bool doResampleField = false;
      for (unsigned int i = 0; i < Dimension; i++ )
	{
	  if (spacingM[i]!=spacing[i])
	    {
	      doResampleField = true;
	      std::cout << "Resampling field 1 relative to mask, spacing ["
					    << niftk::ConvertToString( (int) i ) + "] "
					    << niftk::ConvertToString( spacing[i] ) << " to "
					    << niftk::ConvertToString( spacingM[i] ) << std::endl;
	    }
	  if (sizeM[i]!=size[i])
	    {
	      doResampleField = true;
	      std::cout << "Resampling field 1 relative to mask, size ["
					    << niftk::ConvertToString( (int) i ) << "] "
					    << niftk::ConvertToString( size[i] ) << " to "
					    << niftk::ConvertToString( sizeM[i] ) << std::endl;
	    }
	  if (originM[i]!=origin[i])
	    {
	      doResampleField = true;
	      std::cout << "Resampling field 1 relative to mask, origin ["
					    << niftk::ConvertToString( (int) i ) << "] "
					    << niftk::ConvertToString( origin[i] ) << " to "
					    << niftk::ConvertToString( originM[i] ) << std::endl;
	    }
	  for (unsigned int j = 0; j < Dimension; j++ )
	    {
	      if (directionM[i][j]!=direction[i][j])
		{
		  doResampleField = true;
		  std::cout << "Resampling field 1 relative to mask, direction ["
						<< niftk::ConvertToString( (int) i ) << "]["
						<< niftk::ConvertToString( (int) j ) << "] "
						<< niftk::ConvertToString( direction[i][j] ) << " to "
						<< niftk::ConvertToString( directionM[i][j] ) << std::endl;
		}
	    }
                  
	}
      if (doResampleField)
	{
	  std::cout << "Changing field 1 to image format size and spacing of mask" << std::endl;
	  // resample if necessary
	  fieldResample->SetSize( sizeM );
	  fieldResample->SetOutputOrigin( originM );
	  fieldResample->SetOutputSpacing( spacingM );
	  fieldResample->SetOutputDirection( directionM );
	  fieldResample->SetDefaultPixelValue( zeroDisplacement );
	  fieldResample->SetTransform( rigidIdentityTransform );
                  
	  fieldResample->SetInput(singleField1);
	  fieldResample->Update();
	  singleField1 = fieldResample->GetOutput();
	  singleField1->DisconnectPipeline();

	  region = regionM;
	  size = sizeM;
	  origin = originM;
	  spacing = spacingM;
	  direction = directionM;
	}
    }
  

  // Prepare error images
  // ~~~~~~~~~~~~~~~~~~~~

  initialerrorImage = InternalImageType::New();
  initialerrorImage->SetRegions(region);
  initialerrorImage->SetOrigin(origin);
  initialerrorImage->SetSpacing(spacing);
  initialerrorImage->SetDirection(direction);
  initialerrorImage->Allocate();
  
  errorImage = InternalImageType::New();
  errorImage->SetRegions(region);
  errorImage->SetOrigin(origin);
  errorImage->SetSpacing(spacing);
  errorImage->SetDirection(direction);
  errorImage->Allocate();
  

  // Prepare resampling of deformation field 2 if needed
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  fieldResample->SetSize( size );
  fieldResample->SetOutputOrigin( origin );
  fieldResample->SetOutputSpacing( spacing );
  fieldResample->SetOutputDirection( direction );
  fieldResample->SetDefaultPixelValue( zeroDisplacement );


  // check if resampling is necessary
                  
  FieldRegionType region2 = singleField2->GetLargestPossibleRegion();
  FieldSizeType size2 = singleField2->GetLargestPossibleRegion().GetSize();
  FieldPointType origin2 = singleField2->GetOrigin();
  FieldSpacingType spacing2 = singleField2->GetSpacing();
  FieldDirectionType direction2 = singleField2->GetDirection();
  
  
  bool doResampleField = false;
  for (unsigned int i = 0; i < Dimension; i++ )
    {
      if (spacing2[i]!=spacing[i])
	{
	  doResampleField = true;
	  std::cout << "Resampling field 2 relative to field 1, spacing ["
					<< niftk::ConvertToString( (int) i ) << "] "
					<< niftk::ConvertToString( spacing2[i] ) << " to "
					<< niftk::ConvertToString( spacing[i] ) << std::endl;
	}
      if (size2[i]!=size[i])
	{
	  doResampleField = true;
	  std::cout << "Resampling field 2 relative to field 1, size ["
					<< niftk::ConvertToString( (int) i ) << "] "
					<< niftk::ConvertToString( size2[i] ) << " to "
					<< niftk::ConvertToString( size[i] ) << std::endl;
	}
      if (origin2[i]!=origin[i])
	{
	  doResampleField = true;
	  std::cout << "Resampling field 2 relative to field 1, origin ["
					<< niftk::ConvertToString( (int) i ) << "] "
					<< niftk::ConvertToString( origin2[i] ) << " to "
					<< niftk::ConvertToString( origin[i] ) << std::endl;
	}
      for (unsigned int j = 0; j < Dimension; j++ )
	{
	  if (direction2[i][j]!=direction[i][j])
	    {
	      doResampleField = true;
	      std::cout << "Resampling field 2 relative to field 1, direction ["
					    << niftk::ConvertToString( (int) i ) << "]["
					    << niftk::ConvertToString( (int) j ) << "] "
					    << niftk::ConvertToString( direction2[i][j] ) << " to "
					    << niftk::ConvertToString( direction[i][j] ) << std::endl;
	    }
	}
    }
  if (doResampleField)
    {
      std::cout << "Changing field 2 to image format size and spacing of deformation field 1" << std::endl;
      // resample if necessary
      fieldResample->SetTransform( rigidIdentityTransform );
      fieldResample->SetInput(singleField2);
      fieldResample->Update();
      singleField2 = fieldResample->GetOutput();
      singleField2->DisconnectPipeline();
    }

  //std::cout <<  "size1" << singleField1->GetLargestPossibleRegion().GetSize() << std::endl;
  //std::cout <<  "size2" << singleField2->GetLargestPossibleRegion().GetSize() << std::endl;
  //std::cout <<  "sizeE" << errorImage->GetLargestPossibleRegion().GetSize() << std::endl;
  //if (maskName != NULL)
  //{
  //        std::cout <<  "sizeM" << mask->GetLargestPossibleRegion().GetSize() << std::endl;
  //}
  
  
  // Iterate through computing error
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InternalPixelType sumDiffSq = 0;

  FieldIndexType Findex;
  DisplacementType displacement1;
  DisplacementType displacement2;
  DisplacementType difference;

  InternalPixelType error;
  InternalPixelType meanError = 0.0;
  InternalPixelType maxError = 0.0;
  InternalPixelType minError = 1000000.0;
  InternalPixelType stdError = 0.0;
  
  InternalPixelType initialerror;
  InternalPixelType initialmeanError = 0.0;
  InternalPixelType initialmaxError = 0.0;
  InternalPixelType initialminError = 1000000.0;
  InternalPixelType initialstdError = 0.0;
  
  FieldIterator itField1( singleField1, singleField1->GetLargestPossibleRegion() );
  FieldIterator itField2( singleField2, singleField2->GetLargestPossibleRegion() );

  unsigned int N = 0;
  
  for ( itField1.GoToBegin(), itField2.GoToBegin(); !itField1.IsAtEnd(); ++itField1, ++itField2)
    {
      Findex = itField1.GetIndex();
      
      if ((maskName == NULL) || (mask->GetPixel(Findex) == mask_value))
	{
	  
	  // displacement
	  displacement1 = itField1.Get();                 
	  displacement2 = itField2.Get();
          
	  //std::cout << Findex << " " << displacement1 << " " <<  displacement2 << std::endl;
	  difference = displacement1 - displacement2;
          
	  initialerror = 0.0;
	  error = 0.0;
	  for (unsigned int m=0; m< Dimension; m++ )
	    {
	      initialerror = initialerror + displacement1[m]*displacement1[m];                          
	      error = error + difference[m]*difference[m];                          
	    }
	  initialerror = sqrt(initialerror);
	  error = sqrt(error);
          
	  initialmeanError = initialmeanError + initialerror;
	  meanError = meanError + error;
          
	  if (initialmaxError < initialerror) initialmaxError = initialerror;
	  if (initialminError > initialerror) initialminError = initialerror;
          
	  if (maxError < error) maxError = error;
	  if (minError > error) minError = error;
          
	  N=N+1;
	  //std::cout << N << " " << error << " " << minError << " " << meanError/N << " " <<  maxError << std::endl;
          
	  // always store in image
	  initialerrorImage->SetPixel(Findex, initialerror);
	  errorImage->SetPixel(Findex, error);
	}
    }

  initialmeanError = initialmeanError / N;
  meanError = meanError / N;
  
  //
  // second pass, compute standard deviation
  //
  {
    ImageIterator it( initialerrorImage, initialerrorImage->GetLargestPossibleRegion() );

    sumDiffSq = 0;
    if (maskName != NULL)
      {
	MaskIterator itM( mask, mask->GetLargestPossibleRegion() );
    for ( it.GoToBegin(), itM.GoToBegin(); !it.IsAtEnd(); ++it, ++itM)
	  {
	    if (itM.Get() ==  mask_value)
	      {
		initialerror = it.Get();
		sumDiffSq = sumDiffSq + (initialerror - initialmeanError)*(initialerror - initialmeanError);
	      }
	  }
      }
    else
      {
    for ( it.GoToBegin(); !it.IsAtEnd(); ++it)
	  {
	    initialerror = it.Get();
	    sumDiffSq = sumDiffSq + (initialerror - initialmeanError)*(initialerror - initialmeanError);
	  }
      }
    initialstdError = vcl_sqrt(sumDiffSq / (N - 1));
    
    std::cout <<  "Initial min  error: " << niftk::ConvertToString( initialminError  ) << std::endl;
    std::cout <<  "Initial mean error: " << niftk::ConvertToString( initialmeanError ) << std::endl;
    std::cout <<  "Initial max  error: " << niftk::ConvertToString( initialmaxError  ) << std::endl;
    std::cout <<  "Initial std  error: " << niftk::ConvertToString( initialstdError  ) << std::endl;
  }


  {
    ImageIterator it( errorImage, errorImage->GetLargestPossibleRegion() );

    sumDiffSq = 0;
    if (maskName != NULL)
      {
	MaskIterator itM( mask, mask->GetLargestPossibleRegion() );
    for ( it.GoToBegin(), itM.GoToBegin(); !it.IsAtEnd(); ++it, ++itM)
	  {
	    if (itM.Get() ==  mask_value)
	      {
		error = it.Get();
		sumDiffSq = sumDiffSq + (error-meanError) * (error-meanError);
	      }
	  }
      }
    else
      {
    for ( it.GoToBegin(); !it.IsAtEnd(); ++it)
	  {
	    error = it.Get();
	    sumDiffSq = sumDiffSq + (error-meanError) * (error-meanError);
	  }
      }
    stdError = vcl_sqrt(sumDiffSq / (N - 1));
    
    std::cout <<  "Registration min  error: " << niftk::ConvertToString( minError  ) << std::endl;
    std::cout <<  "Registration mean error: " << niftk::ConvertToString( meanError ) << std::endl;
    std::cout <<  "Registration max  error: " << niftk::ConvertToString( maxError  ) << std::endl;
    std::cout <<  "Registration std  error: " << niftk::ConvertToString( stdError  ) << std::endl;
  }


  // Write the error image to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (outName != NULL)
    {  
      writer->SetFileName( outName );
      writer->SetInput( errorImage );

      std::cout <<  "Writing error image to file: " << outName << std::endl;
          
      try 
	{ 
	  writer->Update(); 
	} 
      catch( itk::ExceptionObject & err ) 
	{ 
	  std::cerr << "ExceptionObject caught !" << std::endl; 
	  std::cerr << err << std::endl; 
	  return EXIT_FAILURE;
	}
    }

  return EXIT_SUCCESS;
}




