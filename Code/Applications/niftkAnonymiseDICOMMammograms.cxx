/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

/*!
 * \file niftkAnonymiseDICOMMammograms.cxx 
 * \page niftkAnonymiseDICOMMammograms
 * \section niftkAnonymiseDICOMMammograms Search for DICOM mammograms in a
 * directory and anonymise them by removing patient information from
 * the DICOM header and/or applying a rectangular mask to remove the
 * label.
 *
 */


#include "FileHelper.h"

#include "itkLogHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkGDCMImageIO.h"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"

#include <vector>

#include "niftkAnonymiseDICOMMammogramsCLP.h"


namespace fs = boost::filesystem;

//  -------------------------------------------------------------------------
//  arguments
//  -------------------------------------------------------------------------

struct arguments
{
  std::string dcmDirectoryIn;
  std::string dcmDirectoryOut;

  bool flgAnonymiseDICOMHeader;
  bool flgAnonymiseImageLabel;

  float labelWidth;
  float labelHeight;

  std::string labelPosition;

  arguments() {

    flgAnonymiseDICOMHeader = false;
    flgAnonymiseImageLabel = false;

    labelWidth  = 0.;
    labelHeight = 0.;

  }
};


// -------------------------------------------------------------------------
// main()
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  arguments args;


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  if ( ( dcmDirectoryIn.length() == 0 ) || ( dcmDirectoryOut.length() == 0 ) )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  args.dcmDirectoryIn  = dcmDirectoryIn;
  args.dcmDirectoryOut  = dcmDirectoryOut;

  args.flgAnonymiseDICOMHeader = flgAnonymiseDICOMHeader;
  args.flgAnonymiseImageLabel  = flgAnonymiseImageLabel;

  args.labelWidth  = labelWidth;
  args.labelHeight = labelHeight;

  args.labelPosition  = labelPosition;


  std::cout << std::endl << "Examining directory: " 
	    << args.dcmDirectoryIn << std::endl << std::endl;


  // Get the list of files in the directory
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::vector< std::string > fileNames;
  niftk::GetRecursiveFileNamesInDirectory( args.dcmDirectoryIn, fileNames );


  // Iterate through each file and anonymise it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::vector< std::string >::iterator iterFileNames;       

  typedef signed short InputPixelType;
  const unsigned int   InputDimension = 2;

  typedef itk::Image< InputPixelType, InputDimension > InputImageType;
  typedef itk::ImageFileReader< InputImageType > ReaderType;
  typedef itk::ImageRegionIterator< InputImageType > IteratorType;  

  ReaderType::Pointer reader = ReaderType::New();

  typedef itk::GDCMImageIO           ImageIOType;
  ImageIOType::Pointer gdcmImageIO = ImageIOType::New();

  reader->SetImageIO( gdcmImageIO );

  for ( iterFileNames = fileNames.begin(); 
	iterFileNames < fileNames.end(); 
	++iterFileNames )
  {
    reader->SetFileName( *iterFileNames );
    
    try
    {
      reader->Update();
    }

    catch (itk::ExceptionObject &ex)
    {
      std::cout << "Skipping file (not DICOM?): " << *iterFileNames << std::endl << std::endl;
      continue;
    }

    std::cout << *iterFileNames << std::endl;


    // Set the label region to zero

    if ( args.flgAnonymiseImageLabel )
    {
      InputImageType::RegionType region;
      InputImageType::SizeType size;
      InputImageType::IndexType start;

      InputImageType::Pointer image = reader->GetOutput();

      region = image->GetLargestPossibleRegion();

      size = region.GetSize();

      std::cout << "Image size: " << size << std::endl;

      start[0] = size[0];
      start[1] = size[1];

      size[0] = static_cast<unsigned int>( static_cast<float>(size[0]) * args.labelWidth/100. );
      size[1] = static_cast<unsigned int>( static_cast<float>(size[1]) * args.labelHeight/100. );

      if ( args.labelPosition == std::string( "Top-Left" ) )
      {
	start[0] = 0;
	start[1] = 0;
      }
      else if ( args.labelPosition == std::string( "Top-Right" ) )
      {
	start[0] -= size[0];
	start[1] = 0;
      }
      else if ( args.labelPosition == std::string( "Bottom-Right" ) )
      {
	start[0] -= size[0];
	start[1] -= size[1];
      }
      else if ( args.labelPosition == std::string( "Bottom-Left" ) )
      {
	start[0] = 0;
	start[1] -= size[1];
      }

      region.SetSize( size );
      region.SetIndex( start );

      std::cout << "Removing label from region: " << region << std::endl;

      IteratorType itLabel( image, region );
  
      for ( itLabel.GoToBegin(); 
	    ! itLabel.IsAtEnd() ; 
	    ++itLabel )
      {
	itLabel.Set( 0 );
      }
    }
    

    // Write the image to the output file

    

  }


  return EXIT_SUCCESS;
}

