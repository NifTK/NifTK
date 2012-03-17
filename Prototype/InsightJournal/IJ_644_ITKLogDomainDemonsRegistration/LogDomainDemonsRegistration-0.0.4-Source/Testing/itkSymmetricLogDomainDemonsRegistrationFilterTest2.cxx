#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <iostream>

#include "itkSymmetricLogDomainDemonsRegistrationFilter.h"

#include "itkImageFileWriter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkVectorCastImageFilter.h"
#include "itkWarpImageFilter.h"



namespace{
  // The following class is used to support callbacks
  // on the filter in the pipeline that follows later
template<typename TRegistration>
class ShowProgressObject
{
public:
  ShowProgressObject(TRegistration* o)
   {m_Process = o;}

  void ShowProgress()
    {
    std::cout << "Progress: " << m_Process->GetProgress() << "  ";
    std::cout << "Iter: " << m_Process->GetElapsedIterations() << "  ";
    std::cout << "Metric: "   << m_Process->GetMetric()   << "  ";
    std::cout << "RMSChange: " << m_Process->GetRMSChange() << "  ";
    std::cout << std::endl;
    if ( m_Process->GetElapsedIterations() == 150 )
      { m_Process->StopRegistration(); }
    }
   
  typename TRegistration::Pointer m_Process;
};
}



// Template function to fill in an image with a circle.
template <class TImage>
void
FillWithCircle(TImage * image,
          double * center,
          double radius,
          typename TImage::PixelType foregnd,
          typename TImage::PixelType backgnd )
{
  typedef itk::ImageRegionIteratorWithIndex<TImage> Iterator;
  Iterator it( image, image->GetBufferedRegion() );
  it.GoToBegin();
    
  typename TImage::IndexType index;
  double r2 = vnl_math_sqr( radius );

  for(; !it.IsAtEnd(); ++it)
    {
    index = it.GetIndex();
    double distance = 0;
    for( unsigned int j = 0; j < TImage::ImageDimension; j++ )
      {
       distance += vnl_math_sqr((double) index[j] - center[j]);
      }
    if( distance <= r2 ) it.Set( foregnd );
    else it.Set( backgnd ); 
    }
}


template < class ImageType >
void WriteImage( const ImageType & input, const std::string & filename )
{
   typedef itk::ImageFileWriter< ImageType > WriterType;
   typename WriterType::Pointer writer = WriterType::New();

   writer->SetInput( &input );
   writer->SetFileName( filename.c_str() );

   try {
      writer->Update();
   }
   catch(itk::ExceptionObject & e)
   {
      std::cerr<<"Writer caught an exception: "<<e.what()<<std::endl;
   }
}



// ----------------------------------------------

int main(int, char* [] )
{
  const unsigned int ImageDimension = 3;

  typedef itk::Vector<float,ImageDimension> VectorType;
  typedef itk::Image<VectorType,ImageDimension> FieldType;
  typedef itk::Image<float,ImageDimension> ImageType;

  typedef FieldType::PixelType  PixelType;
  typedef FieldType::IndexType  IndexType;

  bool testPassed = true;

  
  //--------------------------------------------------------
  std::cout << "Generate input images and initial deformation field";
  std::cout << std::endl;


  // Declare fixed image
  ImageType::RegionType fixed_region;
  ImageType::SizeType fixed_size = {{50, 35, 50}};
  ImageType::IndexType fixed_index;
  fixed_index.Fill( 0 );
  fixed_region.SetSize( fixed_size );
  fixed_region.SetIndex( fixed_index );

  ImageType::DirectionType fixed_direction;
  fixed_direction.SetIdentity();
#if ( defined(ITK_USE_ORIENTED_IMAGE_DIRECTION) && defined(ITK_IMAGE_BEHAVES_AS_ORIENTED_IMAGE) )
  //fixed_direction(1,1)=-1;
#endif

  ImageType::SpacingType fixed_spacing;
  fixed_spacing.Fill( 0.9 );

  ImageType::PointType fixed_origin;
  fixed_origin.Fill( 0.5 );

  ImageType::Pointer fixed = ImageType::New();

  fixed->SetRegions( fixed_region );
  fixed->Allocate();
  fixed->SetDirection( fixed_direction );
  fixed->SetSpacing( fixed_spacing );
  fixed->SetOrigin( fixed_origin );

  // Fill the fixed image with a circle
  itk::Point<double,ImageDimension> center_pt_fixed;
  for (unsigned int i=0; i<ImageDimension; ++i ) {
     center_pt_fixed[i] = fixed_origin[i] + (fixed_size[i] * fixed_spacing[i])/2.0;
  }

  itk::ContinuousIndex<double,ImageDimension> center_cind_fixed;
  fixed->TransformPhysicalPointToContinuousIndex( center_pt_fixed, center_cind_fixed);
  
  const double radius = 8.0;
  const ImageType::PixelType fgnd = 250.0;
  const ImageType::PixelType bgnd = 15.0;
  
  FillWithCircle<ImageType>( fixed, center_cind_fixed.GetDataPointer(),
                             radius/fixed_spacing[0], fgnd, bgnd );

  WriteImage<ImageType>(*fixed.GetPointer(),"fixed.mha");


  // Declare moving image
  ImageType::RegionType moving_region;
  ImageType::SizeType moving_size = {{40, 40, 40}};
  ImageType::IndexType moving_index;
  moving_index.Fill( 0 );
  moving_region.SetSize( moving_size );
  moving_region.SetIndex( moving_index );

  ImageType::DirectionType moving_direction;
  //moving_direction.SetIdentity();
  moving_direction = fixed_direction;

  ImageType::SpacingType moving_spacing;
  moving_spacing.Fill( 1.1 );

  ImageType::PointType moving_origin;
  moving_origin.Fill( 1.5 );

  ImageType::Pointer moving = ImageType::New();

  moving->SetRegions( moving_region );
  moving->Allocate();
  moving->SetDirection( moving_direction );
  moving->SetSpacing( moving_spacing );
  moving->SetOrigin( moving_origin );


  // Fill the moving image with a circle
  itk::ContinuousIndex<double,ImageDimension> center_cind_moving;
  moving->TransformPhysicalPointToContinuousIndex( center_pt_fixed, center_cind_moving);
  for (unsigned int i=0; i<ImageDimension; ++i ) {
     center_cind_moving[i] += 1.0;
  }
  
  FillWithCircle<ImageType>( moving, center_cind_moving.GetDataPointer(),
                             radius/moving_spacing[0], fgnd, bgnd );

  
  WriteImage<ImageType>(*moving.GetPointer(),"moving.mha");

  
   
  //-------------------------------------------------------------

  std::cout << "Run registration and warp moving" << std::endl;

  typedef itk::SymmetricLogDomainDemonsRegistrationFilter<ImageType,ImageType,FieldType> RegistrationType;
  RegistrationType::Pointer registrator = RegistrationType::New();
 
  registrator->SetMovingImage( moving );
  registrator->SetFixedImage( fixed );
  registrator->SetNumberOfIterations( 100 );
  registrator->SetStandardDeviations( 0.7 );
  registrator->SetMaximumUpdateStepLength( 1.0 );
  registrator->SetMaximumError( 0.08 );
  registrator->SetMaximumKernelWidth( 10 );
  registrator->SetIntensityDifferenceThreshold( 0.001 );
  registrator->SetNumberOfBCHApproximationTerms( 2 );

  // Turn on inplace execution
  //registrator->InPlaceOn();
  
  // Progress tracking
  typedef ShowProgressObject<RegistrationType> ProgressType;
  ProgressType progressWatch(registrator);
  itk::SimpleMemberCommand<ProgressType>::Pointer command;
  command = itk::SimpleMemberCommand<ProgressType>::New();
  command->SetCallbackFunction(&progressWatch,
                     &ProgressType::ShowProgress);
  registrator->AddObserver( itk::ProgressEvent(), command);

  try {
     registrator->Update();
     testPassed = false;
  }
  catch (...) {
     std::cout << "Caught expected exception (different sized images is not yet implemented)." << std::endl;
  }


  /*\todo
  // Warper for the moving image
  typedef itk::WarpImageFilter<ImageType,ImageType,FieldType> WarperType;
  WarperType::Pointer warper = WarperType::New();

  // Interpolator
  typedef WarperType::CoordRepType CoordRepType;
  typedef itk::NearestNeighborInterpolateImageFunction<ImageType,CoordRepType>
    InterpolatorType;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  warper->SetInput( moving );
  warper->SetDeformationField( registrator->GetDeformationField() );
  warper->SetInterpolator( interpolator );
  warper->SetOutputSpacing( fixed->GetSpacing() );
  warper->SetOutputOrigin( fixed->GetOrigin() );
  warper->SetOutputDirection( fixed->GetDirection() );
  warper->SetEdgePaddingValue( bgnd );
  
  warper->Print( std::cout );
  
  warper->Update();

  WriteImage<ImageType>(*warper->GetOutput(),"warped-moving.mha");
  

  // ---------------------------------------------------------

  std::cout << "Compare warped moving and fixed." << std::endl;

  itk::ImageRegionIterator<ImageType> fixedIter( fixed,
                                 fixed->GetBufferedRegion() );
  itk::ImageRegionIterator<ImageType> warpedIter( warper->GetOutput(),
                                  fixed->GetBufferedRegion() );


  unsigned int numPixelsDifferent = 0;
  while( !fixedIter.IsAtEnd() )
    {
		if( fixedIter.Get() != warpedIter.Get() )
		  {
			 numPixelsDifferent++;
		  }
		++fixedIter;
		++warpedIter;
    }

  std::cout << "Number of pixels that differ: " << numPixelsDifferent; 
  std::cout << std::endl;

  if( numPixelsDifferent > 100 )
    {
    std::cout << "Test failed - too many pixels differ." << std::endl;
    testPassed = false;
    }
  
  registrator->Print( std::cout );
  */
  
  
  if ( !testPassed )
    {
    std::cout << "Test failed" << std::endl;
    return EXIT_FAILURE;
    }

  std::cout << "Test passed." << std::endl;
  return EXIT_SUCCESS;
}
