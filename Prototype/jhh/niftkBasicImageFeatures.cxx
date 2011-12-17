
/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: jhh                 $
 $Date:: 2011-12-16 15:11:16 +#$
 $Rev:: 8052                   $

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
#include "CommandLineParser.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkBasicImageFeaturesImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImage.h"
#include "itkUnaryFunctorImageFilter.h"
#include "itkScalarToRGBBIFPixelFunctor.h"
#include "itkScalarToRGBOBIFPixelFunctor.h"
#include "itkMaskImageFilter.h"


struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "st", NULL, "Perform single threaded execution [multi-threaded]."},

  {OPT_SWITCH, "orientate", NULL, "Calculate orientated BIFs [no]."},
  {OPT_SWITCH, "n72", NULL, "Calculate orientations in one degree increments [45degs]."},

  {OPT_SWITCH, "vflip", NULL, "Flip the orientation vertically (e.g. for PA vs AP views)."},
  {OPT_SWITCH, "hflip", NULL, "Flip the orientation horizontally (e.g. for ML vs LM views)."},

  {OPT_SWITCH, "lines", NULL, "Only include linear information in the colour image."},

  {OPT_DOUBLEx2, "origin", "ox,oy", "Orientate relative to this origin in mm (0,0 = corner of the image)."},

  {OPT_FLOAT, "sigma", "value", "The Guassian std. dev. in mm at which to compute the BIFs [1.0]."},
  {OPT_FLOAT, "e", "epsilon", "The noise suppression parameter [1e-05]."},

  {OPT_STRING, "u2D", "filename", "Local reference orientation in 'x'."},
  {OPT_STRING, "v2D", "filename", "Local reference orientation in 'y'."},

  {OPT_STRING, "mask", "filename", "Only compute BIFs where the mask image is non-zero."},

  {OPT_STRING, "oS00", "filename", "Save the zero order smoothed image to a file."},
  {OPT_STRING, "oS10", "filename", "Save the first derivative in 'x' to a file."},
  {OPT_STRING, "oS01", "filename", "Save the first derivative in 'y' to a file."},
  {OPT_STRING, "oS11", "filename", "Save the second derivative in 'xy' to a file."},
  {OPT_STRING, "oS20", "filename", "Save the second derivative in 'xx' to a file."},
  {OPT_STRING, "oS02", "filename", "Save the second derivative in 'yy' to a file."},

  {OPT_STRING, "oFlat", "filename", "Save the flatness response to a file."},
  {OPT_STRING, "oSlope", "filename", "Save the slope-like response to a file."},
  {OPT_STRING, "oDarkBlob", "filename", "Save the dark blob response to a file."},
  {OPT_STRING, "oLightBlob", "filename", "Save the light blob response to a file."},
  {OPT_STRING, "oDarkLine", "filename", "Save the dark line response to a file."},
  {OPT_STRING, "oLightLine", "filename", "Save the light line response to a file."},
  {OPT_STRING, "oSaddle", "filename", "Save the saddlelike response to a file."},

  {OPT_STRING, "oVar", "filename", "Save the BIF response variance image to a file."},

  {OPT_STRING, "oh",   "filename", "Write the histogram of BIFs to a file.."},
  {OPT_STRING, "opng", "filename", "Write the label image as a colour PNG file for display purposes."},
  {OPT_STRING, "o",    "filename", "The output label image."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "The input image."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to compute basic image features for a 2D image.\n"
  }
};


enum {
  O_SINGLE_THREADED,

  O_ORIENTATE,
  O_72_ORIENTATIONS,

  O_FLIP_VERTICALLY,
  O_FLIP_HORIZONTALLY,
  
  O_DISPLAY_LINES_ONLY,

  O_ORIGIN,

  O_SIGMA_IN_MM,
  O_EPSILON,

  O_ORIENTATION_INX,
  O_ORIENTATION_INY,

  O_MASK,

  O_OUTPUT_S00,
  O_OUTPUT_S10,
  O_OUTPUT_S01,
  O_OUTPUT_S11,
  O_OUTPUT_S20,
  O_OUTPUT_S02,

  O_OUTPUT_FLAT,
  O_OUTPUT_SLOPE,
  O_OUTPUT_DARK_BLOB,
  O_OUTPUT_LIGHT_BLOB,
  O_OUTPUT_DARK_LINE,
  O_OUTPUT_LIGHT_LINE,
  O_OUTPUT_SADDLE,    

  O_OUTPUT_VARIANCE,

  O_OUTPUT_HISTOGRAM,
  O_OUTPUT_COLOUR_IMAGE,
  O_OUTPUT_IMAGE,

  O_INPUT_IMAGE
};


int main( int argc, char *argv[] )
{
  bool flgSingleThreaded;
  bool flgOrientate;

  bool flgFlipVertically;
  bool flgFlipHorizontally;

  bool flgN72;

  bool flgDisplayLinesOnly;

  float sigmaInMM = 1;
  float epsilon = 1.0e-05;

  double *origin = 0;

  std::string fileOrientationInX;
  std::string fileOrientationInY;

  std::string fileMask;

  std::string fileOutputS00;
  std::string fileOutputS10;
  std::string fileOutputS01;
  std::string fileOutputS11;
  std::string fileOutputS20;
  std::string fileOutputS02;

  std::string fileOutputFlat;
  std::string fileOutputSlope;
  std::string fileOutputDarkBlob;
  std::string fileOutputLightBlob;
  std::string fileOutputDarkLine;
  std::string fileOutputLightLine;
  std::string fileOutputSaddle;   

  std::string fileOutputVariance;

  std::string fileOutputHistogram;
  std::string fileOutputImage;
  std::string fileOutputColourImage;

  std::string fileInputImage;
  
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_SINGLE_THREADED, flgSingleThreaded );

  CommandLineOptions.GetArgument( O_ORIENTATE, flgOrientate );

  CommandLineOptions.GetArgument( O_72_ORIENTATIONS, flgN72 );

  CommandLineOptions.GetArgument( O_FLIP_VERTICALLY,   flgFlipVertically );
  CommandLineOptions.GetArgument( O_FLIP_HORIZONTALLY, flgFlipHorizontally );

  CommandLineOptions.GetArgument( O_DISPLAY_LINES_ONLY, flgDisplayLinesOnly);

  CommandLineOptions.GetArgument( O_ORIGIN, origin );

  CommandLineOptions.GetArgument( O_SIGMA_IN_MM, sigmaInMM );
  CommandLineOptions.GetArgument( O_EPSILON, epsilon );

  CommandLineOptions.GetArgument( O_ORIENTATION_INX, fileOrientationInX );
  CommandLineOptions.GetArgument( O_ORIENTATION_INY, fileOrientationInY );

  if ( (fileOrientationInX.length() || fileOrientationInY.length()) && origin) {

    std::cerr <<"Command line options: -u2D and -v2D cannot be used with -origin";
    return EXIT_FAILURE;
  }                
    
  if ( (fileOrientationInX.length() || fileOrientationInY.length()) 
       && ! (fileOrientationInX.length() && fileOrientationInY.length()) ) {

    std::cerr <<"Both command line options: -u2D and -v2D are required";
    return EXIT_FAILURE;
  }                
    
  CommandLineOptions.GetArgument( O_MASK, fileMask );

  CommandLineOptions.GetArgument( O_OUTPUT_S00, fileOutputS00 );
  CommandLineOptions.GetArgument( O_OUTPUT_S10, fileOutputS10 );
  CommandLineOptions.GetArgument( O_OUTPUT_S01, fileOutputS01 );
  CommandLineOptions.GetArgument( O_OUTPUT_S11, fileOutputS11 );
  CommandLineOptions.GetArgument( O_OUTPUT_S20, fileOutputS20 );
  CommandLineOptions.GetArgument( O_OUTPUT_S02, fileOutputS02 );

  CommandLineOptions.GetArgument( O_OUTPUT_FLAT,       fileOutputFlat );
  CommandLineOptions.GetArgument( O_OUTPUT_SLOPE,      fileOutputSlope );
  CommandLineOptions.GetArgument( O_OUTPUT_DARK_BLOB,  fileOutputDarkBlob );
  CommandLineOptions.GetArgument( O_OUTPUT_LIGHT_BLOB, fileOutputLightBlob );
  CommandLineOptions.GetArgument( O_OUTPUT_DARK_LINE,  fileOutputDarkLine );
  CommandLineOptions.GetArgument( O_OUTPUT_LIGHT_LINE, fileOutputLightLine );
  CommandLineOptions.GetArgument( O_OUTPUT_SADDLE,     fileOutputSaddle );

  CommandLineOptions.GetArgument( O_OUTPUT_VARIANCE, fileOutputVariance );
  CommandLineOptions.GetArgument( O_OUTPUT_HISTOGRAM, fileOutputHistogram );
  CommandLineOptions.GetArgument( O_OUTPUT_COLOUR_IMAGE, fileOutputColourImage );
  CommandLineOptions.GetArgument( O_OUTPUT_IMAGE, fileOutputImage );

  CommandLineOptions.GetArgument( O_INPUT_IMAGE, fileInputImage );


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  // Define the dimension of the images
  const unsigned int ImageDimension = 2;

  typedef float InputPixelType;
  typedef itk::Image<InputPixelType, ImageDimension> InputImageType;

  typedef itk::ImageFileReader< InputImageType > FileReaderType;

  typedef float OutputPixelType;
  typedef itk::Image<OutputPixelType, ImageDimension> OutputImageType;

  typedef itk::BasicImageFeaturesImageFilter< InputImageType, OutputImageType > BasicImageFeaturesFilterType;

  typedef BasicImageFeaturesFilterType::MaskImageType MaskImageType;

  typedef itk::ImageFileReader< MaskImageType > MaskReaderType;



  FileReaderType::Pointer imageReader = FileReaderType::New();

  imageReader->SetFileName(fileInputImage);

  try
  { 
    std::cout << "Reading the input image";
    imageReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }


  // Read the local orientation images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  FileReaderType::Pointer xOrientReader, yOrientReader;

  if ( (fileOrientationInX.length() > 0) && (fileOrientationInY.length() > 0) ) {

    flgOrientate = true; 

    xOrientReader = FileReaderType::New();

    xOrientReader->SetFileName( fileOrientationInX );

    try
      { 
	std::cout << "Reading the local orientation in 'x'";
	xOrientReader->Update();
      }
    catch (itk::ExceptionObject &ex)
      { 
	std::cout << ex << std::endl;
	return EXIT_FAILURE;
      }

    yOrientReader = FileReaderType::New();

    yOrientReader->SetFileName( fileOrientationInY );

    try
      { 
	std::cout << "Reading the local orientation in 'y'";
	yOrientReader->Update();
      }
    catch (itk::ExceptionObject &ex)
      { 
	std::cout << ex << std::endl;
	return EXIT_FAILURE;
      }
  }


  // Read the mask
  // ~~~~~~~~~~~~~

  MaskReaderType::Pointer maskReader = 0;

  if ( fileMask.length() > 0 ) {

    maskReader = MaskReaderType::New();

    maskReader->SetFileName(fileMask);

    try
      { 
	std::cout << "Reading the mask image";
	maskReader->Update();
      }
    catch (itk::ExceptionObject &ex)
      { 
	std::cout << ex << std::endl;
	return EXIT_FAILURE;
      }
  }


  // Create the basic image features filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  BasicImageFeaturesFilterType::Pointer BIFsFilter = BasicImageFeaturesFilterType::New();

  if (flgSingleThreaded)
    BIFsFilter->SetSingleThreadedExecution();

  BIFsFilter->SetSigma( sigmaInMM );
  BIFsFilter->SetEpsilon( epsilon );

  if (flgOrientate) {
    BIFsFilter->CalculateOrientatedBIFs();

    if ( flgN72 )
      BIFsFilter->SetNumberOfOrientations( 72 );
  }

  if ( flgFlipVertically )   BIFsFilter->SetFlipVertically();
  if ( flgFlipHorizontally ) BIFsFilter->SetFlipHorizontally();

  if ( flgDisplayLinesOnly ) BIFsFilter->LinesOnly();

  if (origin) {
    BasicImageFeaturesFilterType::OriginType bifOrigin;

    bifOrigin[0] = origin[0];
    bifOrigin[1] = origin[1];

    BIFsFilter->SetOrigin( bifOrigin );
  }

  if ( (fileOrientationInX.length() > 0) && (fileOrientationInY.length() > 0) )
    BIFsFilter->SetLocalOrientation( xOrientReader->GetOutput(), 
				     yOrientReader->GetOutput() );

  if ( maskReader ) 
    BIFsFilter->SetMask( maskReader->GetOutput() );

  BIFsFilter->SetInput( imageReader->GetOutput() );
  
  try
  {
    std::cout << "Computing basic image features";
    BIFsFilter->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    std::cerr << e << std::endl;
  }
  
  
  // Compute a histogram of the BIFs
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputHistogram.length() > 0 ) {

    unsigned int iBin;
    unsigned int nBins;

    float nPixels = 0;
    float *histogram;

    OutputPixelType pixel;

    OutputImageType::Pointer bifs = BIFsFilter->GetOutput();

    if (flgOrientate) {
      if ( flgN72 )
	nBins = 183;
      else
	nBins = 23;
    }
    else
      nBins = 7;

    histogram = new float[nBins];

    for (iBin=0; iBin<nBins; iBin++) 
      histogram[ iBin ] = 0.;
  
    if ( maskReader ) {

      typedef itk::ImageRegionConstIterator< MaskImageType > MaskIteratorType;
  
      MaskImageType::Pointer mask = maskReader->GetOutput();

      MaskIteratorType itMask( mask, mask->GetLargestPossibleRegion() );
    
      MaskImageType::IndexType index;

      itMask.GoToBegin();

      while (! itMask.IsAtEnd() ) {
      
	index = itMask.GetIndex();	

	if ( itMask.Get() > 0 ) {	// if inside the mask

	  pixel = bifs->GetPixel( index );

	  if ( (pixel < 0) || (pixel >= nBins) )
	    std::cerr << "BIFF value ("
					   << niftk::ConvertToString(pixel)
					   << ") exceeds histogram range (0 to "
					   << niftk::ConvertToString(nBins - 1) << ".";

	  else {
	    nPixels++;
	    histogram[ (unsigned int) pixel ]++;
	  }
	}

	++itMask;
      }
    }
    else {
      typedef itk::ImageRegionConstIterator< OutputImageType > IteratorType;
  
      IteratorType itBIFs( bifs, bifs->GetLargestPossibleRegion() );
    
      itBIFs.GoToBegin();

      while (! itBIFs.IsAtEnd() ) {
      
	pixel = itBIFs.Get();

	if ( (pixel < 0) || (pixel >= nBins) )
	  std::cerr <<std::string("BIFF value (")
					 << niftk::ConvertToString(pixel)
					 << ") exceeds histogram range (0 to "
					 << niftk::ConvertToString(nBins - 1) + ".";
      
	else {
	  nPixels++;
	  histogram[ (unsigned int) pixel ]++;
	}

	++itBIFs;
      }

    }

    std::fstream fout;
    fout.open( fileOutputHistogram.c_str(), std::ios::out );

    if ((! fout) || fout.bad()) {
      std::cerr << "Failed to open file: "
				     << fileOutputHistogram.c_str();
      exit(1);
    }
  
    for (iBin=0; iBin<nBins; iBin++) 

      fout << std::setw(6) << iBin << " "
	   << histogram[ iBin ]/nPixels << std::endl;
  
    delete histogram;
    fout.close();
  }
	

  // Write the derivatives?
  // ~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputS00.length() != 0 ) BIFsFilter->WriteDerivativeToFile( 0, fileOutputS00 );
  if ( fileOutputS10.length() != 0 ) BIFsFilter->WriteDerivativeToFile( 1, fileOutputS10 );
  if ( fileOutputS01.length() != 0 ) BIFsFilter->WriteDerivativeToFile( 2, fileOutputS01 );
  if ( fileOutputS11.length() != 0 ) BIFsFilter->WriteDerivativeToFile( 3, fileOutputS11 );
  if ( fileOutputS20.length() != 0 ) BIFsFilter->WriteDerivativeToFile( 4, fileOutputS20 );
  if ( fileOutputS02.length() != 0 ) BIFsFilter->WriteDerivativeToFile( 5, fileOutputS02 );


  // Write the filter responses?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputFlat.length()      != 0 ) BIFsFilter->WriteFilterResponseToFile( 0, fileOutputFlat      );
  if ( fileOutputSlope.length()     != 0 ) BIFsFilter->WriteFilterResponseToFile( 1, fileOutputSlope     );
  if ( fileOutputDarkBlob.length()  != 0 ) BIFsFilter->WriteFilterResponseToFile( 2, fileOutputDarkBlob  );
  if ( fileOutputLightBlob.length() != 0 ) BIFsFilter->WriteFilterResponseToFile( 3, fileOutputLightBlob );
  if ( fileOutputDarkLine.length()  != 0 ) BIFsFilter->WriteFilterResponseToFile( 4, fileOutputDarkLine  );
  if ( fileOutputLightLine.length() != 0 ) BIFsFilter->WriteFilterResponseToFile( 5, fileOutputLightLine );
  if ( fileOutputSaddle.length()    != 0 ) BIFsFilter->WriteFilterResponseToFile( 6, fileOutputSaddle    );


  // Write the BIF image?
  // ~~~~~~~~~~~~~~~~~~~~

  if (fileOutputImage.length() != 0) {

    typedef itk::ImageFileWriter< OutputImageType > FileWriterType;

    FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( fileOutputImage );
    writer->SetInput( BIFsFilter->GetOutput() );

    try
      {
	std::cout << "Writing the BIF output image.";
	writer->Update();
      }
    catch (itk::ExceptionObject &e)
      {
	std::cerr << e << std::endl;
      }
  }

  if (fileOutputColourImage.length() != 0) {

    typedef itk::RGBPixel<unsigned char> RGBPixelType;
    typedef itk::Image<RGBPixelType, 2> RGBImageType;

    typedef itk::ImageFileWriter< RGBImageType > FileWriterType;

    FileWriterType::Pointer writer = FileWriterType::New();
    writer->SetFileName( fileOutputColourImage );

    if (flgOrientate) {
      if ( flgN72 ) {
	typedef itk::Functor::ScalarToRGBOBIFPixelFunctor<OutputPixelType, 72> ColorMapFunctorType;
	typedef itk::UnaryFunctorImageFilter<OutputImageType, RGBImageType, ColorMapFunctorType> ColorMapFilterType;
      
	ColorMapFilterType::Pointer colormapper = ColorMapFilterType::New();
	colormapper->SetInput(BIFsFilter->GetOutput());
	colormapper->UpdateLargestPossibleRegion();
	  
	writer->SetInput(colormapper->GetOutput());
      }
      else {
	typedef itk::Functor::ScalarToRGBOBIFPixelFunctor<OutputPixelType, 8> ColorMapFunctorType;
	typedef itk::UnaryFunctorImageFilter<OutputImageType, RGBImageType, ColorMapFunctorType> ColorMapFilterType;
	
	ColorMapFilterType::Pointer colormapper = ColorMapFilterType::New();
	colormapper->SetInput(BIFsFilter->GetOutput());
	colormapper->UpdateLargestPossibleRegion();
	  
	writer->SetInput(colormapper->GetOutput());
      }
    }
    else {
      typedef itk::Functor::ScalarToRGBBIFPixelFunctor<OutputPixelType> ColorMapFunctorType;
      typedef itk::UnaryFunctorImageFilter<OutputImageType, RGBImageType, ColorMapFunctorType> ColorMapFilterType;
      
      ColorMapFilterType::Pointer colormapper = ColorMapFilterType::New();
      colormapper->SetInput(BIFsFilter->GetOutput());
      colormapper->UpdateLargestPossibleRegion();

      writer->SetInput(colormapper->GetOutput());
    }

    try
      {
	std::cout << "Writing the BIF colour output image.";
	writer->Update();
      }
    catch (itk::ExceptionObject &e)
      {
	std::cerr << e << std::endl;
      }
  }

  
  // Calculate the BIF response variance image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputVariance.length() > 0 ) {

    float mean, variance, S00;
    float rFlat, rSlope, rDarkBlob, rLightBlob, rDarkLine, rLightLine, rSaddle;

    OutputImageType::Pointer imVariance = OutputImageType::New();
    OutputImageType::RegionType region = imageReader->GetOutput()->GetLargestPossibleRegion();

    imVariance->SetRegions( region );
    imVariance->SetSpacing( imageReader->GetOutput()->GetSpacing() );
    imVariance->SetOrigin( imageReader->GetOutput()->GetOrigin() );

    imVariance->Allocate( );
    imVariance->FillBuffer( 0. );

    if ( maskReader ) {

      typedef itk::ImageRegionConstIterator< MaskImageType > MaskIteratorType;
  
      MaskImageType::Pointer mask = maskReader->GetOutput();

      MaskIteratorType itMask( mask, mask->GetLargestPossibleRegion() );
    
      MaskImageType::IndexType index;

      itMask.GoToBegin();

      while (! itMask.IsAtEnd() ) {
      
	index = itMask.GetIndex();	

	if ( itMask.Get() > 0 ) {	// if inside the mask

	  S00 = BIFsFilter->GetS00()->GetPixel( index );

	  rFlat      = BIFsFilter->GetResponseFlat(     )->GetPixel( index );
	  rSlope     = BIFsFilter->GetResponseSlope(    )->GetPixel( index );
	  rDarkBlob  = BIFsFilter->GetResponseDarkBlob( )->GetPixel( index );
	  rLightBlob = BIFsFilter->GetResponseLightBlob()->GetPixel( index );
	  rDarkLine  = BIFsFilter->GetResponseDarkLine( )->GetPixel( index );
	  rLightLine = BIFsFilter->GetResponseLightLine()->GetPixel( index );
	  rSaddle    = BIFsFilter->GetResponseSaddle(   )->GetPixel( index );

	  mean = ( rFlat + rSlope + rDarkBlob + rLightBlob + rDarkLine + rLightLine + rSaddle )/7.;

	  variance = ( (rFlat      - mean)*(rFlat      - mean) + 
		       (rSlope     - mean)*(rSlope     - mean) + 
		       (rDarkBlob  - mean)*(rDarkBlob  - mean) + 
		       (rLightBlob - mean)*(rLightBlob - mean) + 
		       (rDarkLine  - mean)*(rDarkLine  - mean) + 
		       (rLightLine - mean)*(rLightLine - mean) + 
		       (rSaddle    - mean)*(rSaddle    - mean) )/7.;

	  if ( S00 )
	    imVariance->SetPixel( index, variance/S00 );
	  else
	    imVariance->SetPixel( index, 0.);
	}

	++itMask;
      }
    }
    else {
      typedef itk::ImageRegionConstIterator< OutputImageType > IteratorType;
  
      IteratorType itVar( imVariance, imVariance->GetLargestPossibleRegion() );

      OutputImageType::IndexType index;
    
      itVar.GoToBegin();
      
      while (! itVar.IsAtEnd() ) {
	
	index = itVar.GetIndex();	

	S00 = BIFsFilter->GetS00()->GetPixel( index );

	rFlat      = BIFsFilter->GetResponseFlat(     )->GetPixel( index );
	rSlope     = BIFsFilter->GetResponseSlope(    )->GetPixel( index );
	rDarkBlob  = BIFsFilter->GetResponseDarkBlob( )->GetPixel( index );
	rLightBlob = BIFsFilter->GetResponseLightBlob()->GetPixel( index );
	rDarkLine  = BIFsFilter->GetResponseDarkLine( )->GetPixel( index );
	rLightLine = BIFsFilter->GetResponseLightLine()->GetPixel( index );
	rSaddle    = BIFsFilter->GetResponseSaddle(   )->GetPixel( index );

	mean = ( rFlat + rSlope + rDarkBlob + rLightBlob + rDarkLine + rLightLine + rSaddle )/7.;

	variance = ( (rFlat      - mean)*(rFlat      - mean) + 
		     (rSlope     - mean)*(rSlope     - mean) + 
		     (rDarkBlob  - mean)*(rDarkBlob  - mean) + 
		     (rLightBlob - mean)*(rLightBlob - mean) + 
		     (rDarkLine  - mean)*(rDarkLine  - mean) + 
		     (rLightLine - mean)*(rLightLine - mean) + 
		     (rSaddle    - mean)*(rSaddle    - mean) )/7.;
	
	  if ( S00 )
	    imVariance->SetPixel( index, variance/S00 );
	  else
	    imVariance->SetPixel( index, 0.);
	
	++itVar;
      }
    }

    typedef itk::ImageFileWriter< OutputImageType > FileWriterType;

    FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( fileOutputVariance );
    writer->SetInput( imVariance );

    try
      {
	std::cout << "Writing the BIF variance output image.";
	writer->Update();
      }
    catch (itk::ExceptionObject &e)
      {
	std::cerr << e << std::endl;
      }
  }

}
