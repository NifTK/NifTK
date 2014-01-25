/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

#include <iomanip> 

#include <niftkConversionUtils.h>

#include <itkLogHelper.h>
#include <itkImage.h>
#include <itkCommandLineHelper.h>

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include <itkMinimumMaximumImageCalculator.h>
#include <itkImageToHistogramFilter.h>
#include <itkImageMomentsCalculator.h>
#include <itkConnectedThresholdImageFilter.h>

#include <itkMammogramMaskSegmentationImageFilter.h>

#include <niftkMammogramMaskSegmentationCLP.h>

/*!
 * \file niftkMammogramMaskSegmentation.cxx
 * \page niftkMammogramMaskSegmentation
 * \section niftkMammogramMaskSegmentationSummary Segments a mammogram generating a binary mask corresponding to the breast area.
 *
 * This program uses ITK ImageFileReader to load an image, and then uses MammogramMaskSegmentationImageFilter to segment the breast reagion before writing the output with ITK ImageFileWriter.
 *
 * \li Dimensions: 2.
 * \li Pixel type: Scalars only of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float and double.
 *
 * \section niftkMammogramMaskSegmentationCaveats Caveats
 * \li None
 */
struct arguments
{
  std::string inputImage;
  std::string outputImage;  
};

const unsigned int Dimension = 2;
typedef float InputPixelType;
  
typedef itk::Image< InputPixelType, Dimension > InputImageType;   

typedef itk::Statistics::ImageToHistogramFilter< InputImageType > ImageToHistogramFilterType;
 


// -----------------------------------------------------------------------
// WriteHistogramToTextFile()
// -----------------------------------------------------------------------

void WriteHistogramToTextFile( std::string fileName,
                               ImageToHistogramFilterType::HistogramType *histogram )
{
  unsigned int i;
  double modeFreq = 0;
  std::ofstream fout( fileName.c_str() );

  if ((! fout) || fout.bad()) {
    std::cerr << "ERROR: Could not open file: " << fileName << std::endl;
    exit( EXIT_FAILURE );
  }

  for (i=0; i<histogram->Size(); i++)
  {
    if (  histogram->GetFrequency(i) > modeFreq )
    {
      modeFreq = histogram->GetFrequency(i);
    }
  }

  for (i=0; i<histogram->Size(); i++)
  {
    fout << std::setw( 12 ) << histogram->GetMeasurement(i, 0) << " " 
         << std::setw( 12 ) << ((double) histogram->GetFrequency(i))/modeFreq << std::endl;
  }

  fout.close();
}


// -----------------------------------------------------------------------
// WriteDataToTextFile()
// -----------------------------------------------------------------------

void WriteDataToTextFile( std::string fileName,
                          itk::Array< double > &x,
                          itk::Array< double > &y )
{
  unsigned int i;
  std::ofstream fout( fileName.c_str() );

  if ((! fout) || fout.bad()) {
    std::cerr << "ERROR: Could not open file: " << fileName << std::endl;
    exit( EXIT_FAILURE );
  }

  for (i=0; i<x.GetNumberOfElements(); i++)
  {
    fout << std::setw( 12 ) << x[i] << " " 
         << std::setw( 12 ) << y[i] << std::endl;
  }

  fout.close();
}


// -----------------------------------------------------------------------
// Normalise()
// -----------------------------------------------------------------------

void Normalise( itk::Array< double > &y )
{
  unsigned int i;
  double maxValue = 0;

  for (i=0; i<y.GetNumberOfElements(); i++)
  {
    if ( y[i] > maxValue )
    {
      maxValue = y[i];
    }
  }

  for (i=0; i<y.GetNumberOfElements(); i++)
  {
    y[i] = y[i]/maxValue;
  }
}


// -----------------------------------------------------------------------
// ComputeVariances()
// -----------------------------------------------------------------------

void ComputeVariances( int iStart, int iInc,
                       unsigned int nIntensities, 
                       InputPixelType firstIntensity,
                       ImageToHistogramFilterType::HistogramType *histogram,
                       itk::Array< double > &intensity,
                       itk::Array< double > &nPixelsCummulative,
                       itk::Array< double > &sumIntensitiesCummulative,
                       itk::Array< double > &means,
                       itk::Array< double > &variances )
{
  unsigned int i, j;

  intensity.Fill( 0. );
  nPixelsCummulative.Fill( 0. );
  sumIntensitiesCummulative.Fill( 0. );
  means.Fill( 0. );
  variances.Fill( 0 );

  std::cout << "Image histogram " << std::endl;

  i = 0;
  j = iStart;

  intensity[i] = firstIntensity;
  nPixelsCummulative[i] = histogram->GetFrequency( j );
  sumIntensitiesCummulative[i] = histogram->GetFrequency( j )*intensity[i];

  std::cout << std::setw( 6 ) << j
            << " Freq: " << std::setw( 12 ) << histogram->GetFrequency( j ) 
            << " Intensity: " << std::setw( 12 ) << intensity[i] 
            << " N: " << std::setw( 14 ) << nPixelsCummulative[i] 
            << " Sum: " << std::setw( 14 ) << sumIntensitiesCummulative[i] 
            << std::endl;

  for (i++, j+=iInc; i<nIntensities; i++, j+=iInc)
  {
    intensity[i] = intensity[i-1] + iInc;

    nPixelsCummulative[i] = 
      nPixelsCummulative[i-1] + histogram->GetFrequency( j );
    
    sumIntensitiesCummulative[i] = 
      sumIntensitiesCummulative[i-1] 
      + histogram->GetFrequency( j )*intensity[i];

    std::cout << std::setw( 6 ) << j
              << " Freq: " << std::setw( 12 ) << histogram->GetFrequency( j ) 
              << " Intensity: " << std::setw( 12 ) << intensity[i] 
              << " N: " << std::setw( 14 ) << nPixelsCummulative[i] 
              << " Sum: " << std::setw( 14 ) << sumIntensitiesCummulative[i] 
              << std::endl;
  }
 
  std::cout << "Total frequency: " << histogram->GetTotalFrequency() << std::endl;

  // Compute the variances above and below each level

  i = 0;
  j = iStart;

  if ( nPixelsCummulative[i] > 0. )
  {
    means[j] = sumIntensitiesCummulative[i] / nPixelsCummulative[i];
    
    variances[j] = 
      histogram->GetFrequency(j)
      *( intensity[i] - means[j] )
      *( intensity[i] - means[j] );
  }

  std::cout << std::endl << "Variances: " << std::endl;

  std::cout << std::setw( 6 ) << j
            << " Intensity: " << std::setw( 12 ) << intensity[j]
            << " Mean: " << std::setw( 12 ) << means[j]
            << " Var.: " << std::setw( 12 ) << variances[j]
            << std::endl;

  for(i++, j+=iInc; i<nIntensities; i++, j+=iInc)
  {
    
    if ( nPixelsCummulative[i] > 0. )
    {
      means[j] = sumIntensitiesCummulative[i] / nPixelsCummulative[i];
      variances[j] = variances[j-iInc] + histogram->GetFrequency(j)*( intensity[i] - means[j] )*( intensity[i] - means[j] );
    }
    else
    {
      variances[j] = variances[j-iInc];
    }
    
    std::cout << std::setw( 6 ) << j
              << " Intensity: " << std::setw( 12 ) << intensity[j]
              << " Mean: " << std::setw( 12 ) << means[j]
              << " Var.: " << std::setw( 12 ) << variances[j]
              << std::endl;
  }
}



/**
 * \brief Takes the input and segments it using itk::MammogramMaskSegmentationImageFilter
 */

int main(int argc, char** argv)
{
  unsigned int i;

  typedef float OutputPixelType;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;   

  typedef itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;
  typedef itk::MammogramMaskSegmentationImageFilter<InputImageType, OutputImageType> MammogramMaskSegmentationImageFilterType;


  // Parse the command line arguments
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;

  args.inputImage=inputImage.c_str();
  args.outputImage=outputImage.c_str();

  std::cout << "Input image:  " << args.inputImage << std::endl
            << "Output image: " << args.outputImage << std::endl;

  // Validate command line args

  if (args.inputImage.length() == 0 ||
      args.outputImage.length() == 0)
  {
    return EXIT_FAILURE;
  }


  // Check that the input is 2D
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.inputImage);
  if (dims != 2)
  {
    std::cout << "ERROR: Unsupported image dimension" << std::endl;
    return EXIT_FAILURE;
  }
  else if (dims == 2)
  {
    std::cout << "Input is 2D" << std::endl;
  }


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(args.inputImage);

  try {
    imageReader->Update();
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ERROR: Failed to read image: " 
              << args.inputImage << std::endl
              << err << std::endl; 
    return EXIT_FAILURE;
  }                


#if 0

  // Create the segmentation filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  MammogramMaskSegmentationImageFilterType::Pointer filter = MammogramMaskSegmentationImageFilterType::New();

  filter->SetInput(imageReader->GetOutput());
  filter->SetDebug(true);

#else

  // Alternatively find threshold that maximises:
  // ( Variance_breast * N_background / Variance_backgound ) 
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageType::Pointer image = imageReader->GetOutput();

  image->DisconnectPipeline();

  // Calculate the image range

  typedef itk::MinimumMaximumImageCalculator< InputImageType > MinMaxCalculatorType;
  
  MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();

  minMaxCalculator->SetImage( image );
  minMaxCalculator->Compute();
  
  InputPixelType min = minMaxCalculator->GetMinimum();
  InputPixelType max = minMaxCalculator->GetMaximum();

  std::cout << "Image intensity range from: " << min 
            << " to " << max << std::endl;

  unsigned int nIntensities = static_cast< unsigned int >( max - min + 1. );

  // Calculate the image histogram

  const unsigned int MeasurementVectorSize = 1; // Greyscale
 
  ImageToHistogramFilterType::HistogramType::MeasurementVectorType lowerBound( nIntensities );
  lowerBound.Fill( min );
 
  ImageToHistogramFilterType::HistogramType::MeasurementVectorType upperBound( nIntensities );
  upperBound.Fill( max ) ;
 
  ImageToHistogramFilterType::HistogramType::SizeType size(MeasurementVectorSize);
  size.Fill( nIntensities );
 
  ImageToHistogramFilterType::Pointer imageToHistogramFilter = ImageToHistogramFilterType::New();

  imageToHistogramFilter->SetInput( image );
  imageToHistogramFilter->SetHistogramBinMinimum( lowerBound );
  imageToHistogramFilter->SetHistogramBinMaximum( upperBound );
  imageToHistogramFilter->SetHistogramSize( size );

  try {
    imageToHistogramFilter->Update();
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }                
 
  // Calculate the cummulative stats for each level

  ImageToHistogramFilterType::HistogramType 
    *histogram = imageToHistogramFilter->GetOutput();
 
  double nPixels = histogram->GetTotalFrequency();
  double modeFreq=0, modeIntensity;

  for ( i=0; i<nIntensities; i++ )
  {
    if ( histogram->GetFrequency(i) > modeFreq )
    {
      modeFreq =  histogram->GetFrequency(i);
      modeIntensity = min + i;
    }
  }


  itk::Array< double > lowerIntensities( nIntensities );
  itk::Array< double > upperIntensities( nIntensities );

  itk::Array< double > nPixelsCummulativeLower( nIntensities );
  itk::Array< double > nPixelsCummulativeUpper( nIntensities );

  itk::Array< double > lowerSums( nIntensities );
  itk::Array< double > upperSums( nIntensities );

  itk::Array< double > lowerMeans( nIntensities );
  itk::Array< double > upperMeans( nIntensities );

  itk::Array< double > lowerVariances( nIntensities );
  itk::Array< double > upperVariances( nIntensities );

  ComputeVariances( 0, 1, nIntensities, min, histogram, 
                    lowerIntensities, nPixelsCummulativeLower, lowerSums,
                    lowerMeans, lowerVariances );

  ComputeVariances( nIntensities-1, -1, nIntensities, max, histogram, 
                    upperIntensities, nPixelsCummulativeUpper, upperSums,
                    upperMeans,  upperVariances );

  WriteHistogramToTextFile( std::string( "Histogram.txt"), histogram );

  itk::Array< double > lowerStdDeviations( nIntensities );
  itk::Array< double > upperStdDeviations( nIntensities );

  itk::Array< double > lowerIntensityBias( nIntensities );

  double range = max - min;

  for (i=0; i<nIntensities; i++)
  {
    lowerStdDeviations[i] = sqrt( lowerVariances[i] );
    upperStdDeviations[i] = sqrt( upperVariances[i] );

    lowerIntensityBias[i] = 1. - (lowerIntensities[i] - min)/range;
  }

  Normalise( lowerIntensityBias );

  Normalise( nPixelsCummulativeLower );
  Normalise( lowerSums );
  Normalise( lowerMeans ); 
  Normalise( lowerStdDeviations );
  Normalise( lowerVariances );

  WriteDataToTextFile( std::string( "LowerNumberOfPixelsCummulative.txt"), 
                       lowerIntensities, nPixelsCummulativeLower );
  
  WriteDataToTextFile( std::string( "LowerSumOfIntensities.txt"), 
                       lowerIntensities, lowerSums );
  
  WriteDataToTextFile( std::string( "LowerMeanIntensities.txt"), 
                       lowerIntensities, lowerMeans );
  
  WriteDataToTextFile( std::string( "LowerStdDeviations.txt"), 
                       lowerIntensities, lowerStdDeviations );
  
  WriteDataToTextFile( std::string( "LowerVariances.txt"), 
                       lowerIntensities, lowerVariances );
  
  WriteDataToTextFile( std::string( "LowerIntensityBias.txt"), 
                       lowerIntensities, lowerIntensityBias );
  

  double totalSum = nPixelsCummulativeLower[nIntensities-1];

  itk::Array< double > thresholds( nIntensities );

  double maxThreshold = 0., intThreshold = 0;
  double intensity = min;

  for (i=0; i<nIntensities; i++, intensity += 1.)
  {
    thresholds[i] = lowerIntensityBias[i]*(nPixelsCummulativeLower[i] - lowerVariances[i]);

    if ( thresholds[i] > maxThreshold )
    {
      maxThreshold = thresholds[i];
      intThreshold = intensity;
    }
  }

  Normalise( thresholds );

  WriteDataToTextFile( std::string( "Thresholds.txt"), 
                       lowerIntensities, thresholds );
  
  for (i=0; i<nIntensities; i++, intensity += 1.)
  {
    std::cout << std::setw( 6 ) << intensity
              << " " << std::setw( 12 ) << histogram->GetFrequency(i)/modeFreq
              << " " << std::setw( 12 ) << nPixelsCummulativeLower[i]/nPixels
              << " " << std::setw( 12 ) << lowerSums[i]/totalSum
              << " " << std::setw( 12 ) << lowerMeans[i]
              << " " << std::setw( 12 ) << sqrt( lowerVariances[i] )
              << std::endl;
  }

  std::cout << "Threshold: " << intThreshold << std::endl;


  // Find the center of mass of the image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ImageMomentsCalculator<InputImageType> ImageMomentCalculatorType;

  ImageMomentCalculatorType::VectorType com; 

  com.Fill(0.); 

  ImageMomentCalculatorType::Pointer momentCalculator = ImageMomentCalculatorType::New(); 

  momentCalculator->SetImage( image ); 

  momentCalculator->Compute(); 

  com = momentCalculator->GetCenterOfGravity(); 


  // Region grow from the center of mass
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ConnectedThresholdImageFilter< InputImageType, InputImageType > ConnectedFilterType;

  ConnectedFilterType::Pointer connectedThreshold = ConnectedFilterType::New();

  connectedThreshold->SetInput( image );

  connectedThreshold->SetLower( intThreshold  );
  connectedThreshold->SetUpper( max + 1 );

  connectedThreshold->SetReplaceValue( 1000 );

  InputImageType::IndexType  index;
  
  for ( unsigned int j=0; j<Dimension; j++ )
  {
    index[j] = com[j];
  }

  connectedThreshold->SetSeed( index );

  try
  { 
    std::cout << "Region-growing the image background between: "
              << niftk::ConvertToString(intThreshold) << " and "
              << niftk::ConvertToString(max + 1) << "..."<< std::endl;

    connectedThreshold->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }
  


#endif


  // Create the image writer and execute the pipeline
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

  imageWriter->SetFileName(args.outputImage);
  imageWriter->SetInput( connectedThreshold->GetOutput() );
  
  try
  {
    imageWriter->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }                

  return EXIT_SUCCESS;
}
