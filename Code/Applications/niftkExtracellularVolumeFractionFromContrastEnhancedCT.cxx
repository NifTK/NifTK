
/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: $
 $Date:: $
 $Rev:: $

 Copyright (c) UCL : See the file LICENSE.txt in the top level
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <math.h>
#include <float.h>
#include <iomanip>

#include "ConversionUtils.h"

#include "niftkExtracellularVolumeFractionFromContrastEnhancedCTCLP.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkImageFileWriter.h"


/*!
 * \file niftkExtracellularVolumeFractionFromContrastEnhancedCT.cxx
 * \page niftkExtracellularVolumeFractionFromContrastEnhancedCT
 *                                           
 * \section niftkExtracellularVolumeFractionFromContrastEnhancedCT
 * Uses ITK ImageFileReader to load pre- and post-contrast enhanced CT
 * images, along with corresponding region of interest masks
 * for tissue and the blood pool. Using these the program will calculate the extracellular
 * volume fraction and write the result to a text file.
 *                                                                     
 * \li Dimensions: 3                                                 
 * \li Pixel type: All input images are converted to float on input.   
 *                                                                     
 * \section niftkExtracellularVolumeFractionFromContrastEnhancedCT Caveats             
 * \li All images must have the same size, determined by an ITK Region, which checks the Region Size and Index.
 */                                                                                                            


// --------------------------------------------------------------------------
// SetIteratorSliceOrientation()
// --------------------------------------------------------------------------

template < class ImageType >
void SetIteratorSliceOrientation( itk::ImageSliceIteratorWithIndex< ImageType > &iter,
				  std::string sliceOrientation )
{
  if ( sliceOrientation == std::string( "coronal" ) )
  {
    iter.SetFirstDirection( 0 );
    iter.SetSecondDirection( 2 );
  }
  else if ( sliceOrientation == std::string( "sagittal" ) )
  {
    iter.SetFirstDirection( 1 );
    iter.SetSecondDirection( 2 );
  }
  else // axial
  {
    iter.SetFirstDirection( 0 );
    iter.SetSecondDirection( 1 );
  }   
}


// --------------------------------------------------------------------------
// ReadImage()
// --------------------------------------------------------------------------

template <class PixelType, int Dimension>
typename itk::Image<PixelType, Dimension>::Pointer
ReadImage( std::string fileInput )
{
  typedef typename itk::Image< PixelType, Dimension > ImageType;
  typedef typename itk::ImageFileReader< ImageType > ImageFileReaderType;

  typename ImageFileReaderType::Pointer imageReader = ImageFileReaderType::New();


  imageReader->SetFileName( fileInput.c_str() );

  try
  { 
    std::cout << std::endl << "Reading image: " << fileInput << std::endl;
    imageReader->UpdateLargestPossibleRegion();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cerr << "ERROR: Failed to read image: " <<  fileInput
	       << std::endl << ex << std::endl;
    exit( EXIT_FAILURE );
  }

  return imageReader->GetOutput();
}


// --------------------------------------------------------------------------
// main()
// --------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  // Define the dimension of the images
  const unsigned int ImageDimension = 3;
  
  typedef short MaskPixelType;
  typedef itk::Image<MaskPixelType, ImageDimension> MaskImageType;

  MaskImageType::Pointer imTissueMask;
  MaskImageType::Pointer imBloodMask;
  

  typedef float ImagePixelType;
  typedef itk::Image<ImagePixelType, ImageDimension> ImageType;

  ImageType::Pointer imPreContrast;
  ImageType::Pointer imPostContrast;

  typedef itk::ImageFileReader< ImageType > ImageFileReaderType;
  ImageFileReaderType::Pointer imageReader = ImageFileReaderType::New();

  typedef itk::ImageFileReader< MaskImageType > MaskFileReaderType;
  MaskFileReaderType::Pointer maskReader = MaskFileReaderType::New();

  std::ofstream *fout = 0;


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  if ( fileInputPreContrastImage.length() == 0 )
  {
    std::cerr << "ERROR: The pre-contrast image file name is required" << std::endl;
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  if ( fileInputPostContrastImage.length() == 0 )
  {
    std::cerr << "ERROR: The post-contrast image file name is required" << std::endl;
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  if ( fileInputTissueMask.length() == 0 )
  {
    std::cerr << "ERROR: The tissue mask file name is required" << std::endl;
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  if ( fileInputBloodMask.length() == 0 )
  {
    std::cerr << "ERROR: The blood mask file name is required" << std::endl;
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }


  // Open the text output file

  if ( fileOutputECVTextFile.length() != 0 ) {
    fout = new std::ofstream( fileOutputECVTextFile.c_str() );

    if ((! *fout) || fout->bad()) {
      std::cerr << "ERROR: Could not open file: " << fileOutputECVTextFile << std::endl;
      return EXIT_FAILURE;
    }

    fout->precision(16);

    if ( flgSliceBreakdown )
    {
      *fout  << "Slice number, ";
    }

    *fout  << "Number of Tissue Voxels, "
	   << "Number of Blood Voxels, "
      
	   << "Pre-contrast Mean Intensity of Tissue, "
	   << "Post-contrast Mean Intensity of Tissue, "
      
	   << "Pre-contrast Mean Intensity of Blood, "
	   << "Post-contrast Mean Intensity of Blood, "
      
	   << "Extracellular Volume Fraction" << std::endl;
  }


  // Read the input images
  // ~~~~~~~~~~~~~~~~~~~~~

  // Read the tissue mask
  imTissueMask = ReadImage< MaskPixelType, ImageDimension >( fileInputTissueMask );
  imTissueMask->DisconnectPipeline();

  // Read the blood mask
  imBloodMask = ReadImage< MaskPixelType, ImageDimension >( fileInputBloodMask );
  imBloodMask->DisconnectPipeline();

  // Read the pre-contrast image
  imPreContrast = ReadImage< ImagePixelType, ImageDimension >( fileInputPreContrastImage );
  imPreContrast->DisconnectPipeline();

  // Read the post-contrast image
  imPostContrast = ReadImage< ImagePixelType, ImageDimension >( fileInputPostContrastImage );
  imPostContrast->DisconnectPipeline();


  // Check the images are the same size

  if ( ( imTissueMask->GetLargestPossibleRegion().GetSize() 
	 != imBloodMask->GetLargestPossibleRegion().GetSize() ) || 
       ( imPreContrast->GetLargestPossibleRegion().GetSize() 
	 != imPostContrast->GetLargestPossibleRegion().GetSize() ) || 
       ( imTissueMask->GetLargestPossibleRegion().GetSize() 
	 != imPreContrast->GetLargestPossibleRegion().GetSize() ) )
  {
    std::cerr << "ERROR: The masks and images must have identical sizes" << std::endl;
    return EXIT_FAILURE;
  }


  // Compute the breast density
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  unsigned int iSlice = 0;

  float nVoxelsTissue = 0;
  float nVoxelsBlood = 0;

  float meanPreTissue = 0;
  float meanPostTissue = 0;

  float meanPreBlood = 0;
  float meanPostBlood = 0;

  float ecv = 0.;

  itk::ImageSliceIteratorWithIndex< ImageType > 
    iterPreContrast( imPreContrast, imPreContrast->GetLargestPossibleRegion() );
  
  SetIteratorSliceOrientation< ImageType >( iterPreContrast, slice );
  iterPreContrast.GoToBegin( );

  itk::ImageSliceIteratorWithIndex< ImageType > 
    iterPostContrast( imPostContrast, imPostContrast->GetLargestPossibleRegion() );

  SetIteratorSliceOrientation< ImageType >( iterPostContrast, slice );
  iterPostContrast.GoToBegin( );

  itk::ImageSliceIteratorWithIndex< MaskImageType > 
    iterTissueMask( imTissueMask, imTissueMask->GetLargestPossibleRegion() );

  SetIteratorSliceOrientation< MaskImageType > ( iterTissueMask, slice );
  iterTissueMask.GoToBegin( );

  itk::ImageSliceIteratorWithIndex< MaskImageType > 
    iterBloodMask( imBloodMask, imBloodMask->GetLargestPossibleRegion() );

  SetIteratorSliceOrientation< MaskImageType >( iterBloodMask, slice );
  iterBloodMask.GoToBegin( );

   
  // For each slice in the image
  while ( ! iterPreContrast.IsAtEnd() )
  {

    float nSliceVoxelsTissue = 0.;
    float nSliceVoxelsBlood  = 0.;

    float meanSlicePreTissue  = 0.;
    float meanSlicePostTissue = 0.;

    float meanSlicePreBlood  = 0.;
    float meanSlicePostBlood = 0.;


    // For each line in this slice
    while ( ! iterPreContrast.IsAtEndOfSlice() )
    {
      
      // For each voxel in this line
      while ( ! iterPreContrast.IsAtEndOfLine() )
      {

	if ( iterTissueMask.Get() )
	{
	  nSliceVoxelsTissue++;
	  
	  meanSlicePreTissue += iterPreContrast.Get();
	  meanSlicePostTissue += iterPostContrast.Get();
	}
	
	if ( iterBloodMask.Get() )
	{
	  nSliceVoxelsBlood++;
	  
	  meanSlicePreBlood += iterPreContrast.Get();
	  meanSlicePostBlood += iterPostContrast.Get();
	}

	++iterPreContrast;
	++iterPostContrast;
	++iterTissueMask;
	++iterBloodMask;
      }

      iterPreContrast.NextLine();
      iterPostContrast.NextLine();
      iterTissueMask.NextLine();
      iterBloodMask.NextLine();
    }

    nVoxelsTissue += nSliceVoxelsTissue;
    nVoxelsBlood  += nSliceVoxelsBlood;

    meanPreTissue  += meanSlicePreTissue;
    meanPostTissue += meanSlicePostTissue;

    meanPreBlood  += meanSlicePreBlood;
    meanPostBlood += meanSlicePostBlood;


    // Slice stats

    if ( nSliceVoxelsTissue > 0. )
    {
      meanSlicePreTissue  /= nSliceVoxelsTissue;
      meanSlicePostTissue /= nSliceVoxelsTissue;
    }
    
    if ( nSliceVoxelsBlood > 0. )
    {
      meanSlicePreBlood  /= nSliceVoxelsBlood;
      meanSlicePostBlood /= nSliceVoxelsBlood;

      ecv = ( 1. - haematocrit )*( meanSlicePostTissue - meanSlicePreTissue )
	/ ( meanSlicePostBlood - meanSlicePreBlood );
    }

    if ( flgSliceBreakdown &&
	 (( nSliceVoxelsTissue > 0. ) || ( nSliceVoxelsBlood > 0. )) )
    {
      std::cout << "Slice: " << iSlice << std::endl
		<< "Number of tissue voxels: " << nSliceVoxelsTissue << std::endl
		<< "Number of blood voxels:  " << nSliceVoxelsBlood << std::endl
		<< std::endl
	
		<< "Mean pre-contrast for tissue:  " << meanSlicePreTissue << std::endl
		<< "Mean post-contrast for tissue: " << meanSlicePostTissue << std::endl
		<< std::endl
	
		<< "Mean pre-contrast for blood:  " << meanSlicePreBlood << std::endl
		<< "Mean post-contrast for blood: " << meanSlicePostBlood << std::endl
		<< std::endl
	
		<< "Extracellular volume fraction: " << ecv << std::endl
		<< std::endl;

      if ( fout )
      {
	*fout  << iSlice << ", "
	  
	       << nSliceVoxelsTissue << ", "
	       << nSliceVoxelsBlood << ", "
	  
	       << meanSlicePreTissue << ", "
	       << meanSlicePostTissue << ", "
	  
	       << meanSlicePreBlood << ", "
	       << meanSlicePostBlood << ", "
	  
	       << ecv << std::endl;
      }
    }

    iterPreContrast.NextSlice();
    iterPostContrast.NextSlice();
    iterTissueMask.NextSlice();
    iterBloodMask.NextSlice();

    iSlice++;
  }

  // Global stats

  if ( nVoxelsTissue > 0 )
  {
    meanPreTissue /= nVoxelsTissue;
    meanPostTissue /= nVoxelsTissue;
  }

  if ( nVoxelsBlood > 0 )
  {
    meanPreBlood /= nVoxelsBlood;
    meanPostBlood /= nVoxelsBlood;
  }

  ecv = (1. - haematocrit)*(meanPostTissue - meanPreTissue)/(meanPostBlood - meanPreBlood);

  std::cout << "Number of tissue voxels: " << nVoxelsTissue << std::endl
	    << "Number of blood voxels: " << nVoxelsBlood << std::endl
	    << std::endl

	    << "Mean pre-contrast for tissue: " << meanPreTissue << std::endl
	    << "Mean post-contrast for tissue: " << meanPostTissue << std::endl
	    << std::endl

	    << "Mean pre-contrast for blood: " << meanPreBlood << std::endl
	    << "Mean post-contrast for blood: " << meanPostBlood << std::endl
	    << std::endl

	    << "Extracellular volume fraction: " << ecv << std::endl
	    << std::endl;


  if ( fout )
  {
    if ( flgSliceBreakdown )
    {
      *fout  << "Totals, ";
    }

    *fout  << nVoxelsTissue << ", "
	   << nVoxelsBlood << ", "
      
	   << meanPreTissue << ", "
	   << meanPostTissue << ", "
      
	   << meanPreBlood << ", "
	   << meanPostBlood << ", "
      
	   << ecv << std::endl;
    
    fout->close();
    delete fout;
 
    std::cout << "ECV fraction written to file: " 
	      << fileOutputECVTextFile << std::endl << std::endl;
  }

  
  // Calculate the ECV map
  // ~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputECVImage.length() != 0 )
  {
    float bloodContrast = meanPostBlood - meanPreBlood;

    if ( bloodContrast == 0. )
    {
      std::cerr << "ERROR: There is no contrast between pre- and post-blood pools" << std::endl;
      return EXIT_FAILURE;    
    }

    
    typedef itk::ImageRegionIterator< ImageType > IteratorType;  

    IteratorType iterPre( imPreContrast, imPreContrast->GetLargestPossibleRegion() );
    IteratorType iterPost( imPostContrast, imPostContrast->GetLargestPossibleRegion() );

        
    for ( iterPre.GoToBegin(), iterPost.GoToBegin(); 
	  ! iterPre.IsAtEnd();
	  ++iterPre, ++iterPost )
    {
      ecv = ( 1. - haematocrit )*( iterPost.Get() - iterPre.Get() ) / bloodContrast;
      
      iterPre.Set( ecv );
    }


    typedef itk::ImageFileWriter< ImageType > ImageWriterType;
                                                            
    ImageWriterType::Pointer imageWriter = ImageWriterType::New();

    imageWriter->SetFileName( fileOutputECVImage );                                        
    imageWriter->SetInput( imPreContrast );                                        
                                                                                     
    try                                                                                
    {                                                                                  
      imageWriter->Update();                                                           
    }                                                                                  
    catch( itk::ExceptionObject & err )                                                
    {                                                                                  
      std::cerr << "ERROR: Failed to write output image to file: " << fileOutputECVImage
		<< std::endl << err << std::endl;                                     
      return EXIT_FAILURE;                                                             
    }                                                                                  
 
    std::cout << "ECV fraction map written to image: " 
	      << fileOutputECVImage << std::endl << std::endl;

  }


  return EXIT_SUCCESS;
}
