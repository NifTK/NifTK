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
#include <iostream>
#include "itkBinaryMask3DMeshSource.h"
#include "itkDeformableMesh3DFilter.h"
#include "itkGradientRecursiveGaussianImageFilter.h"
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkImage.h"
#include "itkMesh.h"
#include "itkCovariantVector.h"
#include "itkPointSetToImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTriangleMeshToBinaryImageFilter.h"

int main( int argc, char *argv[] )
{

  if( argc < 4 )
  {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " InputImage  BinaryImage DeformedMaskImage foreground sigma timeStep iterations" << std::endl;
    return 1;
  }
 
  const unsigned int Dimension = 3;
  typedef   double          PixelType;
  typedef itk::Image<PixelType, Dimension>      ImageType;
  typedef itk::Image< short, Dimension >   BinaryImageType;
  typedef  itk::Mesh<double>     MeshType;
  typedef itk::CovariantVector< double, Dimension >  GradientPixelType;
  typedef itk::Image< GradientPixelType, Dimension > GradientImageType;
  typedef itk::GradientMagnitudeRecursiveGaussianImageFilter<ImageType,ImageType> GradientMagnitudeFilterType;
  typedef itk::GradientRecursiveGaussianImageFilter<ImageType, GradientImageType> GradientFilterType;
  typedef itk::BinaryMask3DMeshSource< BinaryImageType, MeshType >  MeshSourceType;
  typedef itk::DeformableMesh3DFilter<MeshType,MeshType>  DeformableFilterType;
  typedef itk::ImageFileReader< ImageType       >  ReaderType;
  typedef itk::ImageFileReader< BinaryImageType >  BinaryReaderType;
  
  ReaderType::Pointer       imageReader   =  ReaderType::New();
  BinaryReaderType::Pointer maskReader    =  BinaryReaderType::New();
  imageReader->SetFileName( argv[1] );
  maskReader->SetFileName(  argv[2] );
  
  GradientMagnitudeFilterType::Pointer  gradientMagnitudeFilter = GradientMagnitudeFilterType::New();
  gradientMagnitudeFilter->SetInput( imageReader->GetOutput() ); 
  gradientMagnitudeFilter->SetSigma( atof(argv[5]) );
  
  GradientFilterType::Pointer gradientMapFilter = GradientFilterType::New();
  gradientMapFilter->SetInput( gradientMagnitudeFilter->GetOutput());
  gradientMapFilter->SetSigma(atof(argv[5]));
  try
  {
    gradientMapFilter->Update();
    std::cout << "The gradient map created!" << std::endl;
  }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "Exception caught when updating gradientMapFilter " << std::endl;
    std::cerr << e << std::endl;
    return -1;
  }

  MeshSourceType::Pointer meshSource = MeshSourceType::New();
  DeformableFilterType::Pointer deformableModelFilter = DeformableFilterType::New();
  deformableModelFilter->SetGradient( gradientMapFilter->GetOutput() );
  BinaryImageType::Pointer mask = maskReader->GetOutput();
  meshSource->SetInput( mask );
  meshSource->SetObjectValue(atoi(argv[4]));

  std::cout << "Creating mesh..." << std::endl;
  try 
  {
    meshSource->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception Caught !" << std::endl;
    std::cerr << excep << std::endl;
  }

  deformableModelFilter->SetInput(  meshSource->GetOutput() );
  meshSource->GetOutput()->Print(std::cout);
  std::cout << "Deformable mesh created using Marching Cube!" << std::endl;

  typedef itk::CovariantVector<double, 2>           double2DVector;
  typedef itk::CovariantVector<double, 3>           double3DVector;

  double2DVector stiffness;
  stiffness[0] = 0.1;
  //stiffness[1] = 0.1;
  stiffness[1] = 0.1;

  double3DVector scale;
  scale[0] = 1.0;
  scale[1] = 1.0; 
  scale[2] = 1.0;

  deformableModelFilter->SetStiffness( stiffness );
  deformableModelFilter->SetScale( scale );
  deformableModelFilter->SetTimeStep(atof(argv[6]));
  deformableModelFilter->SetStepThreshold(atoi(argv[7]));
  deformableModelFilter->SetGradientMagnitude( 1.0 );
  std::cout << "Deformable mesh fitting...";

  try 
  {
    deformableModelFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception Caught !" << std::endl;
    std::cerr << excep << std::endl;
  }
  std::cout << "Mesh Source: " << meshSource;

  typedef itk::PointSetToImageFilter<MeshType,ImageType> MeshFilterType;
  MeshFilterType::Pointer meshFilter = MeshFilterType::New();
  meshFilter->SetOrigin(mask->GetOrigin());
  meshFilter->SetSize(mask->GetLargestPossibleRegion().GetSize());
  meshFilter->SetSpacing(mask->GetSpacing());
  meshFilter->SetInput(deformableModelFilter->GetOutput());
  try 
  {
    meshFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception Caught !" << std::endl;
    std::cerr << excep << std::endl;
  }
  
  typedef itk::TriangleMeshToBinaryImageFilter<MeshType,ImageType> MeshToImageFilterType;   
  MeshToImageFilterType::Pointer meshToImageFilter = MeshToImageFilterType::New(); 
  meshToImageFilter->SetInput(deformableModelFilter->GetOutput()); 
  meshToImageFilter->SetOrigin(mask->GetOrigin());
  meshToImageFilter->SetSize(mask->GetLargestPossibleRegion().GetSize());
  meshToImageFilter->SetSpacing(mask->GetSpacing());
  
  typedef itk::ImageFileWriter<ImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(meshToImageFilter->GetOutput());
  writer->SetFileName(argv[3]);
  writer->Update();

  return EXIT_SUCCESS;
}
