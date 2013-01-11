/*=============================================================================

  NifTK: An image processing toolkit jointly developed by the
  Dementia Research Centre, and the Centre For Medical Image Computing
  at University College London.

  See:        http://dementia.ion.ucl.ac.uk/
  http://cmic.cs.ucl.ac.uk/
  http://www.ucl.ac.uk/

  Last Changed      : $Date: 2011-09-20 20:35:56 +0100 (Tue, 20 Sep 2011) $
  Revision          : $Revision: 7340 $
  Last modified by  : $Author: ad $

  Original author   : j.hipwell@ucl.ac.uk

  Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notices for more information.

  ============================================================================*/


#include "itkCommandLineHelper.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkVotingBinaryIterativeHoleFillingImageFilter.h"

#include "niftkVotingBinaryIterativeHoleFillingImageFilterCLP.h"

/*!
 * \file niftkVotingBinaryIterativeHoleFillingImageFilter.cxx
 * \page niftkVotingBinaryIterativeHoleFillingImageFilter
 * \section niftkVotingBinaryIterativeHoleFillingImageFilterSummary 
 *  The \doxygen{VotingBinaryIterativeHoleFillingImageFilter} applies a voting
 *  operation in order to fill-in cavities. This can be used for smoothing
 *  contours and for filling holes in binary images. This filter runs
 *  internally a \doxygen{VotingBinaryHoleFillingImageFilter} until no
 *  pixels change or the maximum number of iterations has been reached.
 *
 * \li Dimensions: 2,3
 *
 * \section niftkVotingBinaryIterativeHoleFillingImageFilter Caveats
 *
 * \li Input image is assumed to be binary
 *
 */


typedef struct arguments
{
  std::string fileInputImage;
  std::string fileOutputImage;

  unsigned int radius;
  unsigned int majority;
  unsigned int numberOfIterations;

  arguments() {
    radius = 1;
    majority = 2;
    numberOfIterations = 100;
  }

} Arguments;


// -------------------------------------------------------------------------------------
// RunHoleFillingFilter(Arguments args)
// -------------------------------------------------------------------------------------

template < const int dimension, class PixelType >
bool RunHoleFillingFilter(Arguments args)
{

  typedef itk::Image< PixelType, dimension >   ImageType;

  typedef itk::ImageFileReader< ImageType  >  ReaderType;
  typedef itk::ImageFileWriter< ImageType >  WriterType;

  typename ReaderType::Pointer reader = ReaderType::New();
  typename WriterType::Pointer writer = WriterType::New();

  reader->SetFileName( args.fileInputImage );
  writer->SetFileName( args.fileOutputImage );


  typedef itk::VotingBinaryIterativeHoleFillingImageFilter< ImageType >  FilterType;

  typename FilterType::Pointer filter = FilterType::New();

  //  The size of the neighborhood is defined along every dimension by
  //  passing a \code{SizeType} object with the corresponding values. The
  //  value on each dimension is used as the semi-size of a rectangular
  //  box. For example, in $2D$ a size of \(1,2\) will result in a $3 \times
  //  5$ neighborhood.

  typename ImageType::SizeType indexRadius;

  indexRadius.Fill( args.radius );

  filter->SetRadius( indexRadius );

  //  Since the filter is expecting a binary image as input, we must specify
  //  the levels that are going to be considered background and foreground. This
  //  is done with the \code{SetForegroundValue()} and

  filter->SetBackgroundValue(   0 );
  filter->SetForegroundValue( 255 );

  //  We must also specify the majority threshold that is going to be used as
  //  the decision criterion for converting a background pixel into a
  //  foreground pixel. The rule of conversion is that a background pixel will
  //  be converted into a foreground pixel if the number of foreground
  //  neighbors surpass the number of background neighbors by the majority
  //  value. For example, in a 2D image, with neighborhood or radius 1, the
  //  neighborhood will have size $3 \times 3$. If we set the majority value to
  //  2, then we are requiring that the number of foreground neighbors should
  //  be at least (3x3 -1 )/2 + majority.

  filter->SetMajorityThreshold( args.majority );

  //  Finally we specify the maximum number of iterations that this filter
  //  should be run. The number of iteration will determine the maximum size of
  //  holes and cavities that this filter will be able to fill-in. The more
  //  iterations you ran, the larger the cavities that will be filled in.

  filter->SetMaximumNumberOfIterations( args.numberOfIterations );

  filter->SetInput( reader->GetOutput() );
  writer->SetInput( filter->GetOutput() );

  writer->Update();

  unsigned int iterationsUsed = filter->GetCurrentNumberOfIterations();

  std::cout << "The filter used " << iterationsUsed << " iterations " << std::endl;
  
  unsigned int numberOfPixelsChanged = filter->GetNumberOfPixelsChanged();

  std::cout << "and changed a total of " << numberOfPixelsChanged << " pixels" << std::endl;

  return EXIT_SUCCESS;
}


// -------------------------------------------------------------------------------------
// main(int argc, char **argv)
// -------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
  int result;


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  Arguments args; 

  PARSE_ARGS;

  if ( fileInputImage.length() == 0 || fileOutputImage.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  args.radius  = radius;
  args.majority  = majority;
  args.numberOfIterations  = numberOfIterations;

  args.fileInputImage  = fileInputImage;
  args.fileOutputImage = fileOutputImage;

  try
  {
      
    std::cout << "Input             :\t" << args.fileInputImage << std::endl;
    std::cout << "Output            :\t" << args.fileOutputImage << std::endl;

    unsigned int nDimensions = itk::PeekAtImageDimensionFromSizeInVoxels( args.fileInputImage );

    if ( (nDimensions != 2) && (nDimensions != 3) )
    {
      std::cout << "Unsupported image dimension" << std::endl;
      return EXIT_FAILURE;
    }

    switch ( itk::PeekAtComponentType( args.fileInputImage ) )
    {
    case itk::ImageIOBase::UCHAR:
      if (nDimensions == 2)
      {
	result = RunHoleFillingFilter<2, unsigned char>(args);  
      }
      else
      {
	result = RunHoleFillingFilter<3, unsigned char>(args);
      }
      break;
    case itk::ImageIOBase::CHAR:
      if (nDimensions == 2)
      {
	result = RunHoleFillingFilter<2, char>(args);  
      }
      else
      {
	result = RunHoleFillingFilter<3, char>(args);
      }
      break;
    case itk::ImageIOBase::USHORT:
      if (nDimensions == 2)
      {
	result = RunHoleFillingFilter<2, unsigned short>(args);  
      }
      else
      {
	result = RunHoleFillingFilter<3, unsigned short>(args);
      }
      break;
    case itk::ImageIOBase::SHORT:
      if (nDimensions == 2)
      {
	result = RunHoleFillingFilter<2, short>(args);  
      }
      else
      {
	result = RunHoleFillingFilter<3, short>(args);
      }
      break;
    case itk::ImageIOBase::UINT:
      if (nDimensions == 2)
      {
	result = RunHoleFillingFilter<2, unsigned int>(args);  
      }
      else
      {
	result = RunHoleFillingFilter<3, unsigned int>(args);
      }
      break;
    case itk::ImageIOBase::INT:
      if (nDimensions == 2)
      {
	result = RunHoleFillingFilter<2, int>(args);  
      }
      else
      {
	result = RunHoleFillingFilter<3, int>(args);
      }
      break;
    case itk::ImageIOBase::ULONG:
      if (nDimensions == 2)
      {
	result = RunHoleFillingFilter<2, unsigned long>(args);  
      }
      else
      {
	result = RunHoleFillingFilter<3, unsigned long>(args);
      }
      break;
    case itk::ImageIOBase::LONG:
      if (nDimensions == 2)
      {
	result = RunHoleFillingFilter<2, long>(args);  
      }
      else
      {
	result = RunHoleFillingFilter<3, long>(args);
      }
      break;
    case itk::ImageIOBase::FLOAT:
      if (nDimensions == 2)
      {
	result = RunHoleFillingFilter<2, float>(args);  
      }
      else
      {
	result = RunHoleFillingFilter<3, float>(args);
      }
      break;
    case itk::ImageIOBase::DOUBLE:
      if (nDimensions == 2)
      {
	result = RunHoleFillingFilter<2, double>(args);  
      }
      else
      {
	result = RunHoleFillingFilter<3, double>(args);
      }
      break;
    default:
      std::cerr << "Unsupported pixel format" << std::endl;
      result = EXIT_FAILURE;
    }
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }                

  return result;
}


