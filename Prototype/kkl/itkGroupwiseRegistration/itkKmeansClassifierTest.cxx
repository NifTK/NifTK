/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 18:04:05 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3325 $
 Last modified by  : $Author: mjc $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkSimpleKMeansClusteringImageFilter.h"

int main( int argc, char * argv [] )
{
  if( argc < 5 )
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0];
    std::cerr << " inputScalarImage inputMask outputLabeledImage contiguousLabels";
    std::cerr << " numberOfClasses mean1 mean2... meanN " << std::endl;
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
  
  //const unsigned int useNonContiguousLabels = atoi( argv[4] );
  //simpleKMeansClusteringImageFilter->SetUseNonContiguousLabels( useNonContiguousLabels );
  
  const unsigned int numberOfInitialClasses = atoi( argv[5] );
  SimpleKMeansClusteringImageFilterType::ParametersType initialMeans(numberOfInitialClasses);
  SimpleKMeansClusteringImageFilterType::ParametersType finalMeans(numberOfInitialClasses);
  SimpleKMeansClusteringImageFilterType::ParametersType finalStds(numberOfInitialClasses);
  
  const unsigned int argoffset = 6;

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


