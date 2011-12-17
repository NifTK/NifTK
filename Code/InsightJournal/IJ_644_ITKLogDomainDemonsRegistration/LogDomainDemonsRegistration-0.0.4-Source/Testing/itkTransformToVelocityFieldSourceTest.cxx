#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <iostream>

#include "itkAffineTransform.h"
#include "itkExponentialDeformationFieldImageFilter2.h"
#include "itkResampleImageFilter.h"
#include "itkTransformToVelocityFieldSource.h"


int main(int, char* [] )
{
  /** Typedefs. */
  const unsigned int Dimension = 2;
  typedef float                            ScalarPixelType;
  typedef double                           CoordRepresentationType;

  typedef itk::Vector<
    ScalarPixelType, Dimension>            VectorPixelType;

  typedef itk::Image<
    VectorPixelType, Dimension>            VelocityFieldType;

  typedef itk::Image<
    VectorPixelType, Dimension>            DeformationFieldType;

  typedef itk::TranslationTransform<
    CoordRepresentationType, Dimension>    TranslationTransformType;

  typedef itk::AffineTransform<
    CoordRepresentationType, Dimension>    AffineTransformType;

  typedef AffineTransformType::ParametersType
                                           ParametersType;

  typedef itk::TransformToVelocityFieldSource<
    VelocityFieldType,
    CoordRepresentationType>               VelocityFieldGeneratorType;

  typedef itk::ExponentialDeformationFieldImageFilter<
    VelocityFieldType,
    DeformationFieldType>                  ExponentialFieldFilterType;

  typedef VelocityFieldType::SizeType      SizeType;
  typedef VelocityFieldType::SpacingType   SpacingType;
  typedef VelocityFieldType::PointType     PointType;
  typedef VelocityFieldType::IndexType     IndexType;
  typedef VelocityFieldType::RegionType    RegionType;
  typedef VelocityFieldType::DirectionType DirectionType;

  /** Create image. */
  SizeType size;
  size.Fill( 24 );
  IndexType index;
  index.Fill(0);
  SpacingType spacing;
  spacing[0] = 1.2;
  spacing[1] = 0.8;
  PointType origin;
  origin[0] = -5;
  origin[1] = 8;
  DirectionType direction;
  direction[0][0] = 0;
  direction[0][1] = -1;

  direction[1][0] = 1;
  direction[1][1] = 0;

  RegionType region;
  region.SetSize(size);
  region.SetIndex(index);

  /** Create transforms. */
  AffineTransformType::Pointer affinetransform = AffineTransformType::New();

  /** Set the options. */
  PointType center;
  center[0] = origin[0] + spacing[0]*size[0]/2.0;
  center[1] = origin[1] + spacing[1]*size[1]/2.0;
  affinetransform->SetCenter( center );

  /** Create and set parameters. */
  ParametersType parameters( affinetransform->GetNumberOfParameters() );
  parameters[0] =  1.1;
  parameters[1] =  0.1;
  parameters[2] =  -0.3;
  parameters[3] =  0.9;
  
  parameters[4] =  -2;
  parameters[5] =  3;
  
  affinetransform->SetParameters( parameters );

  /** Create and setup velocity field generator. */
  VelocityFieldGeneratorType::Pointer velGenerator
    = VelocityFieldGeneratorType::New();
  velGenerator->SetOutputSize( size );
  velGenerator->SetOutputSpacing( spacing );
  velGenerator->SetOutputOrigin( origin );
  velGenerator->SetOutputIndex( index );
  velGenerator->SetOutputDirection( direction );
  velGenerator->SetTransform( affinetransform );
  try
    {
    velGenerator->Update();
    }
  catch ( itk::ExceptionObject & e )
    {
    std::cerr << "Exception detected while generating velocity field";
    std::cerr << " : "  << e.GetDescription();
    return EXIT_FAILURE;
    }

  /** Create and setup exponential filter. */
  ExponentialFieldFilterType::Pointer exponentiator
     = ExponentialFieldFilterType::New();
  exponentiator->SetInput( velGenerator->GetOutput() );
  try
    {
    exponentiator->Update();
    }
  catch ( itk::ExceptionObject & e )
    {
    std::cerr << "Exception detected while generating deformation field";
    std::cerr << " : "  << e.GetDescription();
    return EXIT_FAILURE;
    }

  DeformationFieldType::ConstPointer deffield = exponentiator->GetOutput();
  
  /** Compare the results. */
  typedef itk::ImageRegionConstIteratorWithIndex<DeformationFieldType>  IteratorType;
  IteratorType it( deffield, region );

  PointType pt;
  double diff = 0.0;

  it.GoToBegin();
  while( !it.IsAtEnd() )
    {
    deffield->TransformIndexToPhysicalPoint(it.GetIndex(), pt);
    
    const PointType warppt = pt + it.Value();
    const PointType trsfpt = affinetransform->TransformPoint(pt);

    //std::cout << it.GetIndex()<< " - " << pt << ": "
    //          << trsfpt << " => " << warppt << std::endl;
    
    diff += (warppt-trsfpt).GetSquaredNorm();
    
    ++it;
    }

  diff = vcl_sqrt(diff/region.GetNumberOfPixels());

  std::cout<<"Mean error (affine): "<<diff<<std::endl;

  if ( diff >= 0.3 )
    {
    std::cout<<"Test failed"<<std::endl;
    return EXIT_FAILURE;
    }


  /* Redo the test with a simple translation transform. */
  TranslationTransformType::Pointer translationtransform = TranslationTransformType::New();
  parameters = ParametersType( translationtransform->GetNumberOfParameters() );
  parameters[0] =  -15;
  parameters[1] =  30;
  translationtransform->SetParameters( parameters );
  velGenerator->SetTransform( translationtransform );
  try
    {
    exponentiator->Update();
    }
  catch ( itk::ExceptionObject & e )
    {
    std::cerr << "Exception detected while generating deformation field";
    std::cerr << " : "  << e.GetDescription();
    return EXIT_FAILURE;
    }

  deffield = exponentiator->GetOutput();
  
  it = IteratorType( deffield, region );

  diff = 0.0;

  it.GoToBegin();
  while( !it.IsAtEnd() )
    {
    deffield->TransformIndexToPhysicalPoint(it.GetIndex(), pt);
    
    const PointType warppt = pt + it.Value();
    const PointType trsfpt = translationtransform->TransformPoint(pt);

    //std::cout << it.GetIndex()<< " - " << pt << ": "
    //          << trsfpt << " => " << warppt << std::endl;
    
    diff += (warppt-trsfpt).GetSquaredNorm();
    
    ++it;
    }

  diff = vcl_sqrt(diff/region.GetNumberOfPixels());

  std::cout<<"Mean error (translation): "<<diff<<std::endl;

  if ( diff >= 0.01 )
    {
    std::cout<<"Test failed"<<std::endl;
    return EXIT_FAILURE;
    }

  std::cout<<"Test passed"<<std::endl;
  return EXIT_SUCCESS;
}
