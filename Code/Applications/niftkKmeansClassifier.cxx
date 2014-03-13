/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <itkLogHelper.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkSimpleKMeansClusteringImageFilter.h"

int main( int argc, char * argv [] )
{
  if( argc < 5 )
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cerr);
    std::cerr << "Usage: ";
    std::cerr << argv[0];
    std::cerr << " inputScalarImage inputMask outputLabeledImage numberOfClasses mean1 mean2... meanN ";
    std::cerr << std::endl;
    std::cerr << "Arguments: " << std::endl;
    std::cerr << " inputScalarImage [file]: input image" << std::endl;
    std::cerr << " inputMask [file]: input mask" << std::endl;
    std::cerr << " outputLabeledImage [file]: output image" << std::endl;
    std::cerr << " numberOfClasses [number]: number of classes" << std::endl;
    std::cerr << " mean1 mean2... meanN [numbers]: initial values" << std::endl;
    return EXIT_FAILURE;
  }

  const char* inputImageFileName = argv[1];
  const char* inputMaskFilename = argv[2];
  const char* outputImageFileName = argv[3];
  typedef short  PixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image<PixelType, Dimension > ImageType;
  typedef itk::Image<float, Dimension > FloatImageType;
  typedef itk::ImageFileReader< FloatImageType > ReaderType;
  typedef itk::ImageFileReader< ImageType > MaskReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( inputImageFileName );
  reader->Update(); 
  MaskReaderType::Pointer maskReader = MaskReaderType::New();
  maskReader->SetFileName(inputMaskFilename); 
  maskReader->Update();
  
  typedef itk::SimpleKMeansClusteringImageFilter< FloatImageType, ImageType, ImageType > SimpleKMeansClusteringImageFilterType;
  SimpleKMeansClusteringImageFilterType::Pointer simpleKMeansClusteringImageFilter = SimpleKMeansClusteringImageFilterType::New();
  
  const unsigned int numberOfInitialClasses = atoi( argv[4] );
  SimpleKMeansClusteringImageFilterType::ParametersType initialMeans(numberOfInitialClasses);
  SimpleKMeansClusteringImageFilterType::ParametersType finalMeans(numberOfInitialClasses);
  SimpleKMeansClusteringImageFilterType::ParametersType finalStds(numberOfInitialClasses);
  
  const unsigned int argoffset = 5;

  if( static_cast<unsigned int>(argc) < numberOfInitialClasses + argoffset )
  {
    std::cerr << "Error: " << std::endl;
    std::cerr << numberOfInitialClasses << " classes has been specified ";
    std::cerr << "but no enough means have been provided in the command ";
    std::cerr << "line arguments " << std::endl;
    return EXIT_FAILURE;
  }

  for( unsigned k=0; k < numberOfInitialClasses; k++ )
  {
    const double userProvidedInitialMean = atof( argv[k+argoffset] );
    initialMeans[k] = userProvidedInitialMean;
  }
  
  simpleKMeansClusteringImageFilter->SetInitialMeans(initialMeans);
  simpleKMeansClusteringImageFilter->SetNumberOfClasses(numberOfInitialClasses);  
  simpleKMeansClusteringImageFilter->SetInput(reader->GetOutput());
  simpleKMeansClusteringImageFilter->SetInputMask(maskReader->GetOutput());
  simpleKMeansClusteringImageFilter->Update();
  finalMeans = simpleKMeansClusteringImageFilter->GetFinalMeans();
  finalStds = simpleKMeansClusteringImageFilter->GetFinalStds();
  
  typedef itk::ImageFileWriter<ImageType> WriterType;
  WriterType::Pointer imageWriter = WriterType::New();
  imageWriter->SetInput(simpleKMeansClusteringImageFilter->GetOutput());
  imageWriter->SetFileName(outputImageFileName);
  imageWriter->Update();

  for ( unsigned int i = 0 ; i < numberOfInitialClasses ; ++i )
  {
    std::cout << finalMeans[i] << " " << finalStds[i] << " "; 
  }
  std::cout << std::endl;

  return EXIT_SUCCESS;
  
}


