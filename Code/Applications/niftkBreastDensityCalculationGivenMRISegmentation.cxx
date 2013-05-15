/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <math.h>
#include <float.h>
#include <iomanip>

#include <ConversionUtils.h>

#include <niftkBreastDensityCalculationGivenMRISegmentationCLP.h>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>

/*!
 * \file niftkBreastDensityCalculationGivenMRISegmentation.cxx
 * \page niftkBreastDensityCalculationGivenMRISegmentation
 *                                           
 * \section niftkBreastDensityCalculationGivenMRISegmentationSummary niftkBreastDensityCalculationGivenMRISegmentation
 *
 * Uses ITK ImageFileReader to load a breast mask and glandular tissue segmentation, calculates the breast density fraction and writes the result to a text file.
 *                                                                     
 * \li Dimensions: 3                                                 
 * \li Pixel type: All input images are converted to float on input.   
 *                                                                     
 * \section niftkBreastDensityCalculationGivenMRISegmentationCaveats Caveats             
 * \li All images must have the same size, determined by an ITK Region, which checks the Region Size and Index.
 */                                                                                                            



// --------------------------------------------------------------------------
// main()
// --------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  // Define the dimension of the images
  const unsigned int ImageDimension = 3;
  
  typedef short MaskPixelType;
  typedef itk::Image<MaskPixelType, ImageDimension> MaskImageType;

  typedef itk::ImageFileReader< MaskImageType > MaskFileReaderType;
  
  MaskImageType::Pointer imMask;
  

  typedef float SegmPixelType;
  typedef itk::Image<SegmPixelType, ImageDimension> SegmImageType;

  typedef itk::ImageFileReader< SegmImageType > SegmFileReaderType;
  
  SegmImageType::Pointer imSegmentation;


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  if ( fileInputMask.length() == 0 || fileInputSegmentation.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }


  // Read the input images
  // ~~~~~~~~~~~~~~~~~~~~~

  // Read the breast mask

  MaskFileReaderType::Pointer maskReader = MaskFileReaderType::New();

  maskReader->SetFileName( fileInputMask.c_str() );

  try
  { 
    std::cout << std::endl << "Reading mask image: " << fileInputMask << std::endl;
    maskReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cerr << "ERROR: reading mask image: " <<  fileInputMask
	       << std::endl << ex << std::endl;
    return EXIT_FAILURE;
  }

  imMask = maskReader->GetOutput();
  imMask->DisconnectPipeline();

  // Read the breast segmentation

  SegmFileReaderType::Pointer segmReader = SegmFileReaderType::New();

  segmReader->SetFileName( fileInputSegmentation.c_str() );

  try
  { 
    std::cout << "Reading segmentation image: " << fileInputSegmentation << std::endl << std::endl;
    segmReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cerr << "ERROR: reading segmentation image: " <<  fileInputSegmentation
	       << std::endl << ex << std::endl;
    return EXIT_FAILURE;
  }

  imSegmentation = segmReader->GetOutput();
  imSegmentation->DisconnectPipeline();

  // Check the images are the same size


  if ( imMask->GetLargestPossibleRegion().GetSize() 
       != imSegmentation->GetLargestPossibleRegion().GetSize() )
  {
    std::cerr << "ERROR: The mask and segmentation images have different sizes" << std::endl;
    return EXIT_FAILURE;
  }


  // Compute the breast density
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  float nLeftVoxels = 0;
  float nRightVoxels = 0;

  float totalDensity = 0.;
  float leftDensity = 0.;
  float rightDensity = 0.;

  itk::ImageRegionIteratorWithIndex< MaskImageType > 
    maskIterator( imMask, imMask->GetLargestPossibleRegion() );

  itk::ImageRegionConstIterator< SegmImageType > 
    segmIterator( imSegmentation, imSegmentation->GetLargestPossibleRegion() );

  SegmImageType::SpacingType spacing = imSegmentation->GetSpacing();

  float voxelVolume = spacing[0]*spacing[1]*spacing[2];

  MaskImageType::RegionType region;
  region = imMask->GetLargestPossibleRegion();

  MaskImageType::SizeType lateralSize;
  lateralSize = region.GetSize();
  lateralSize[0] = lateralSize[0]/2;

  MaskImageType::IndexType idx;
   
  for ( maskIterator.GoToBegin(), segmIterator.GoToBegin();
	! maskIterator.IsAtEnd();
	++maskIterator, ++segmIterator )
  {
    if ( maskIterator.Get() )
    {
      idx = maskIterator.GetIndex();

      // Left breast

      if ( idx[0] < (int) lateralSize[0] )
      {
	nLeftVoxels++;
	leftDensity += segmIterator.Get();
      }

      // Right breast

      else 
      {
	nRightVoxels++;
	rightDensity += segmIterator.Get();
      }

      // Both breasts

      totalDensity += segmIterator.Get();
    }
  }
  
  float leftBreastVolume = nLeftVoxels*voxelVolume;
  float rightBreastVolume = nRightVoxels*voxelVolume;

  leftDensity /= nLeftVoxels;
  rightDensity /= nRightVoxels;
  totalDensity /= ( nLeftVoxels + nRightVoxels);

  std::cout << "Number of left breast voxels: " << nLeftVoxels << std::endl
	    << "Volume of left breast: " << leftBreastVolume << " mm^3" << std::endl
	    << "Density of left breast (fraction of glandular tissue): " << leftDensity 
	    << std::endl << std::endl

	    << "Number of right breast voxels: " << nRightVoxels << std::endl
	    << "Volume of right breast: " << rightBreastVolume << " mm^3" << std::endl
	    << "Density of right breast (fraction of glandular tissue): " << rightDensity 
	    << std::endl << std::endl

	    << "Total number of breast voxels: " 
	    << nLeftVoxels + nRightVoxels << std::endl
	    << "Total volume of both breasts: " 
	    << leftBreastVolume + rightBreastVolume << " mm^3" << std::endl
	    << "Combined density of both breasts (fraction of glandular tissue): " 
	    << totalDensity << std::endl << std::endl;


  if ( fileOutputDensityTextFile.length() != 0 ) {
    std::ofstream fout( fileOutputDensityTextFile.c_str() );

    fout.precision(16);

    if ((! fout) || fout.bad()) {
      std::cerr << "ERROR: Could not open file: " << fileOutputDensityTextFile << std::endl;
      return EXIT_FAILURE;
    }

    fout << "Number of left breast voxels, "
	 << "Volume of left breast (mm^3), "
	 << "Density of left breast (fraction of glandular tissue), "
      
	 << "Number of right breast voxels, "
	 << "Volume of right breast (mm^3), "
	 << "Density of right breast (fraction of glandular tissue), "
      
	 << "Total number of breast voxels, "
	 << "Total volume of both breasts (mm^3), "
	 << "Combined density of both breasts (fraction of glandular tissue)" 
	 << std::endl;

    fout << nLeftVoxels << ", "
	 << leftBreastVolume << ", "
	 << leftDensity << ", "
      
	 << nRightVoxels << ", "
	 << rightBreastVolume << ", "
	 << rightDensity << ", "
      
	 << nLeftVoxels + nRightVoxels << ", "
	 << leftBreastVolume + rightBreastVolume << ", "
	 << totalDensity << std::endl;
    
    fout.close();

    std::cout << "Density measurements written to file: " 
	      << fileOutputDensityTextFile << std::endl << std::endl;
  }

  return EXIT_SUCCESS;
}
