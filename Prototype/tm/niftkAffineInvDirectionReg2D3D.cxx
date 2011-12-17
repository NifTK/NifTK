/*=========================================================================

 This program is a modified version of the typical itk registration pipeline.
 It is used to perform a 2D - 3D registration between the MR breast volume
 and the X-ray image.

=========================================================================*/

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegistrationMethod.h"
#include "itkAffineTransform2D3D.h"
#include "itkInvRayCastInterpolateImageFunction.h"
#include "itkInvResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkInvNormalizedCorrelationImageToImageMetric.h" 
#include "itkLinearInterpolateImageFunction.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkDataObject.h"
#include "itkDataObjectDecorator.h"
#include "itkCommand.h"
#include "itkNormalVariateGenerator.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <math.h>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ofstream myFile;

bool preserveVolume;

// FiniteDifferenceNCCImageToImageMetric
namespace itk
{
  template < class TFixedImage, class TMovingImage > 
  class ITK_EXPORT FiniteDifferenceNCCImageToImageMetric : 
    public InvNormalizedCorrelationImageToImageMetric<TFixedImage, TMovingImage>
  {
  public :

    /** Standard class typedefs. */
    typedef FiniteDifferenceNCCImageToImageMetric    Self;
    typedef InvNormalizedCorrelationImageToImageMetric<TFixedImage, TMovingImage >  Superclass;

    typedef SmartPointer<Self>         Pointer;
    typedef SmartPointer<const Self>   ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);
		
    /** Run-time type information (and related methods). */
    itkTypeMacro(FiniteDifferenceNCCImageToImageMetric, NormalizedCorrelationImageToImageMetric);
 
    /** Types transferred from the base class */
    typedef typename Superclass::RealType                 RealType;
    typedef typename Superclass::TransformType            TransformType;
    typedef typename Superclass::TransformPointer         TransformPointer;
    typedef typename Superclass::TransformParametersType  TransformParametersType;
    typedef typename Superclass::TransformJacobianType    TransformJacobianType;
    typedef typename Superclass::GradientPixelType        GradientPixelType;
    typedef typename Superclass::GradientImageType        GradientImageType;
    typedef typename Superclass::InputPointType           InputPointType;
    typedef typename Superclass::OutputPointType          OutputPointType;

    typedef typename Superclass::MeasureType              MeasureType;
    typedef typename Superclass::DerivativeType           DerivativeType;
    typedef typename Superclass::FixedImageType           FixedImageType;
    typedef typename Superclass::MovingImageType          MovingImageType;
    typedef typename Superclass::FixedImageConstPointer   FixedImageConstPointer;
    typedef typename Superclass::MovingImageConstPointer  MovingImageConstPointer;

    /**  Get value and derivatives for multiple valued optimizers. */
    void GetValueAndDerivative( const TransformParametersType & parameters,
				MeasureType& value, DerivativeType& derivative ) const
    {
      //Delta value
      float delta = 0.001; //1 when I changed it

      value = this->GetValue( parameters );
      std::cout << "The metric value is: " << value << std::endl;

      TransformParametersType testPoint;
      testPoint = parameters;

      const unsigned int numberOfParameters = this->GetNumberOfParameters();
      derivative = DerivativeType( numberOfParameters );

      for( unsigned int i=0; i<numberOfParameters; i++) 
	{
	  testPoint[i] -= delta;
	  const MeasureType valuep0 = this->GetValue( testPoint );
	  testPoint[i] += 2 * delta;
	  const MeasureType valuep1 = this->GetValue( testPoint );
	  derivative[i] = (valuep1 - valuep0 ) / ( 2 * delta );
	  testPoint[i] = parameters[i];
	  std::cout << "derivative["<<i<<"] : "<<derivative[i]<<std::endl;
	}
    };

  };
};
// AffineVolumePreservingRegStepOptimizer
namespace itk
{
  class ITK_EXPORT AffineVolumePreservingRegStepOptimizer : 
    public RegularStepGradientDescentOptimizer
  {
  public:
    /** Standard class typedefs. */
    typedef AffineVolumePreservingRegStepOptimizer      Self;
    typedef RegularStepGradientDescentOptimizer         Superclass;
    typedef SmartPointer<Self>                          Pointer;
    typedef SmartPointer<const Self>                    ConstPointer;
  
    /** Method for creation through the object factory. */
    itkNewMacro(Self);
  
    /** Run-time type information (and related methods). */
    itkTypeMacro( AffineVolumePreservingRegStepOptimizer, 
		  RegularStepGradientDescentOptimizer );

    /** Cost function typedefs. */
    typedef Superclass::CostFunctionType        CostFunctionType;
    typedef CostFunctionType::Pointer           CostFunctionPointer;
  
  protected:
    void StepAlongGradient( double factor, const DerivativeType & transformedGradient )
    {
      itkDebugMacro(<<"factor = " << factor << "  transformedGradient= " << transformedGradient );

      const unsigned int spaceDimension = m_CostFunction->GetNumberOfParameters();

      //Updated parameter values according to the optimizer's suggestions
      ParametersType newPosition( spaceDimension );
      ParametersType currentPosition = this->GetCurrentPosition();

      //std::cout << "Calculating new position..." << std::endl;
      //std::cout << "The space dimension is: " << spaceDimension << std::endl;

      for(unsigned int j=0; j<spaceDimension; j++)
	{
	  newPosition[j] = currentPosition[j] + transformedGradient[j] * factor;
	}
      
      if ( preserveVolume )
	{
	  //std::cout << "Calculating new position, according to the volume constraint..." << std::endl;     
	  // new position of the scale parameters along Y and Z axis is specified in order to 
	  // preserve the volume.
	  newPosition[4] = 1./(newPosition[5]*newPosition[3]);// 1./sqrt(newPosition[3]);
	  //newPosition[5] = 1./sqrt(newPosition[3]);
	}

      itkDebugMacro(<<"new position = " << newPosition );

      this->SetCurrentPosition( newPosition );
    };

  };
};


// Class to handle the observer
class CommandIterationUpdate : public itk::Command
{
public:
  typedef CommandIterationUpdate Self;
  typedef itk::Command Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  itkNewMacro( Self );
 
protected:
  CommandIterationUpdate() {};

public:
  typedef itk::AffineVolumePreservingRegStepOptimizer OptimizerType; //RegularStepGradientDescentOptimizer OptimizerType;
  typedef const OptimizerType *OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event)
  {
    Execute( (const itk::Object *)caller, event);
  }

  void Execute(const itk::Object * object, const itk::EventObject & event)
  {
    OptimizerPointer optimizer = dynamic_cast< OptimizerPointer >( object );
    if( ! itk::IterationEvent().CheckEvent( &event ) )
      {
	return;
      }
    myFile << optimizer->GetCurrentIteration() << " = ";
    myFile << optimizer->GetValue() << " : ";
    myFile << optimizer->GetCurrentPosition() << std::endl;
    std::cout << optimizer->GetCurrentIteration() << " = ";
    std::cout << optimizer->GetValue() << " : ";
    std::cout << optimizer->GetCurrentPosition() << std::endl;
  }
};



int main( int argc, char * argv[] )
{
 if( argc < 10 ) 
 { 
   std::cerr << "Usage: " << std::endl;
   std::cerr << argv[0] << "registration  fixedImageFile  movingImageFile  outputImageFile outputParametersFile xstart ystart xsize ysize preserveVolume";
   std::cerr <<  std::endl;
   return EXIT_FAILURE;
 }

 // values used for the DRR generation (see itkDRR).
 float sid = 660.;
 /////////////////////////////////////////

 // input and output decl  
 const int dimension = 3;
 typedef float PixelType; 
 typedef float VolumePixelType; 

 typedef itk::Image< PixelType, dimension > FixedImageType;
 typedef itk::Image< VolumePixelType, dimension > MovingImageType; 
 typedef itk::Image< PixelType, dimension > OutputImageType; 
 
 // reader and writer for the input and output images
 typedef itk::ImageFileReader< FixedImageType >  FixedReaderType;
 typedef itk::ImageFileReader< MovingImageType >  MovingReaderType;
 typedef itk::ImageFileWriter< OutputImageType >  WriterType;

 FixedReaderType::Pointer  fixedImageReader = FixedReaderType::New();
 MovingReaderType::Pointer movingImageReader = MovingReaderType::New();
 WriterType::Pointer writer = WriterType::New();

 fixedImageReader->SetFileName( argv[1] );
 movingImageReader->SetFileName( argv[2] );
 writer->SetFileName( argv[3] );

 preserveVolume = atoi( argv[9] );

 // tranform
 typedef itk::AffineTransform2D3D< double, 3 >  TransformType;
 //TransformType::Pointer transform = TransformType::New();
 TransformType::Pointer regTransform = TransformType::New();

 // optimizer definition
 typedef itk::AffineVolumePreservingRegStepOptimizer OptimizerType; 
 OptimizerType::Pointer optimizer = OptimizerType::New();
 
 // metric type
 typedef itk::FiniteDifferenceNCCImageToImageMetric< FixedImageType, MovingImageType > MetricType;
 MetricType::Pointer metric = MetricType::New();
 
 // interpolator type to evaluate intensities at non-grid positions
 typedef itk::InvRayCastInterpolateImageFunction< MovingImageType, double > InterpolatorType;
 InterpolatorType::Pointer interpolator = InterpolatorType::New();

 // registration method
 typedef itk::ImageRegistrationMethod< FixedImageType, MovingImageType > RegistrationType;
 RegistrationType::Pointer registration = RegistrationType::New();

 // set the parameters of the registration
 registration->SetTransform(  regTransform );
 registration->SetOptimizer( optimizer );
 registration->SetMetric( metric );
 registration->SetInterpolator( interpolator );
 registration->SetFixedImage( fixedImageReader->GetOutput() );
 registration->SetMovingImage( movingImageReader->GetOutput() );

 // update the fixedImage to obtain the buffered area that will be used as the
 // region for the metric to be computed 
 FixedImageType::SizeType size;
 size[0] = atoi( argv[7] );
 size[1] = atoi( argv[8] );
 size[2] = 1;

 // set the region for the input and the output images 
 FixedImageType::RegionType inputRegion;
 inputRegion.SetSize( size );
 FixedImageType::RegionType::IndexType inputStart;
 inputStart[0] = atoi( argv[5] );
 inputStart[1] = atoi( argv[6] );
 inputStart[2] = 0;
 inputRegion.SetIndex( inputStart );

 fixedImageReader->Update();
 registration->SetFixedImageRegion ( inputRegion );

 // set the origin for the 3D volume
 double origin3D[ dimension ];

 const itk::Vector<double, 3> resolution3D = movingImageReader->GetOutput()->GetSpacing();

 typedef MovingImageType::RegionType  ImageRegionType3D;
 typedef ImageRegionType3D::SizeType  SizeType3D;

 ImageRegionType3D region3D = movingImageReader->GetOutput()->GetBufferedRegion();
 SizeType3D        size3D   = region3D.GetSize();

 origin3D[0] = resolution3D[0]*((double) size3D[0]-1)/2.; 
 origin3D[1] = resolution3D[1]*((double) size3D[1]-1)/2.; 
 origin3D[2] = resolution3D[2]*((double) size3D[2]-1)/2.;

 movingImageReader->Update();
 std::cout <<"3D origin: "<<"["<<movingImageReader->GetOutput()->GetOrigin()[0]<<" ,"<<movingImageReader->GetOutput()->GetOrigin()[1]<<" ,"<<movingImageReader->GetOutput()->GetOrigin()[2]<<"]"<< std::endl;

 // initial parameters of the regtransform  
 /*TransformType::OutputVectorType translation; 
 translation[0] = 20;
 translation[1] = 0;
 translation[2] = 0;
 regTransform->Translate( translation );*/
 

 // initial parameters of the regTransform  
 /*TransformType::InputPointType center; 
 center[0] = origin3D[0];
 center[1] = origin3D[1];
 center[2] = origin3D[2];

 regTransform->SetCenter(center);*/


 // set the origin for the 2D image
 double origin2D[ dimension ];

 const itk::Vector<double, 3> resolution2D = fixedImageReader->GetOutput()->GetSpacing();
 typedef FixedImageType::RegionType      ImageRegionType2D;
 typedef ImageRegionType2D::SizeType  SizeType2D;

 ImageRegionType2D region2D = fixedImageReader->GetOutput()->GetBufferedRegion();
 SizeType2D        size2D   = region2D.GetSize();

 origin2D[0] = - resolution2D[0]*((double) size2D[0] - 1.)/2.; 
 origin2D[1] = - resolution2D[1]*((double) size2D[0] - 1.)/2.;  
 origin2D[2] = + 180. ; //sid/2.; 
 
 fixedImageReader->GetOutput()->SetOrigin( origin2D );

 std::cout <<"2D origin: "<<"["<<origin2D[0]<<" ,"<<origin2D[1]<<" ,"<<origin2D[2]<<"]"<< std::endl;

 // Initialisation of the interpolator
 InterpolatorType::InputPointType focalpoint;

 focalpoint[0]= 0;
 focalpoint[1]= 0;
 focalpoint[2]= - (sid-180.);

 std::cout << "Focal point: " << focalpoint << std::endl;

 interpolator->SetFocalPoint(focalpoint);
 interpolator->SetTransform(regTransform);

 // Initialise the registration
 registration->SetInitialTransformParameters( regTransform->GetParameters() );

 //subtract the mean to create more steap valeys
 metric->SetSubtractMean(true);
 
 optimizer->MaximizeOff();
 optimizer->SetMaximumStepLength( 1.00 ); // 4.00
 optimizer->SetMinimumStepLength( 0.001 ); //0.01;
 optimizer->SetNumberOfIterations( 100 );//400 );
 optimizer->SetRelaxationFactor( 0.8 );
 
 // Optimizer weightings 
 itk::Optimizer::ScalesType weightings( regTransform->GetNumberOfParameters() );

 weightings[0] = 1.0;//10.0;//1000.0;
 weightings[1] = 1.0;//1.0;//10000.0; 
 weightings[2] = 1.0;//1.0;//10000.0;
 weightings[3] = 1.0;//1.0;//10000.0;
 weightings[4] = 1.0;//1.0;//1000.0;
 weightings[5] = 1.0;//1.0;//10000.0;
 weightings[6] = 1.0;//10.0;//10000.0;
 weightings[7] = 1.0;//10.0;//10000.0;
 weightings[8] = 1.0;//10.0;//1000.0;
 weightings[9] = 0.001;//0.001;//1.0;
 weightings[10] = 0.001;//0.001;//1.0;
 weightings[11] = 0.001;//0.001;//1.0;

 optimizer->SetScales( weightings );

 myFile.open( argv[4] );

 myFile << "0 = 0 :";
 myFile <<  regTransform->GetParameters()  << std::endl;

 std::cout << "Initial Parameters" << " : "; 
 std::cout <<  regTransform->GetParameters()  << std::endl;

 // Create the observers
 CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
 optimizer->AddObserver( itk::IterationEvent(), observer );

 try
 {
   std::cout << "Before updating registration ... " << std::endl;
   registration->StartRegistration(); //Update(); 
 }
 catch( itk::ExceptionObject & err )
 {
   std::cerr << "ExceptionObject caught !" << std::endl;
   std::cerr << err << std::endl;
   return -1;
 }

 // get the result of the registration
 registration->GetOutput()->Get()->Print(std::cout);
 
 std::cout << "The stopCondition is: " << optimizer->GetStopCondition() << std::endl;

 myFile.close();

 // resampler to use the matrix of the registration result
 typedef itk::InvResampleImageFilter< MovingImageType, FixedImageType > ResampleFilterType;
 ResampleFilterType::Pointer resampler = ResampleFilterType::New();
 movingImageReader->Update();
 resampler->SetInput( movingImageReader->GetOutput() );
 resampler->SetTransform( registration->GetOutput()->Get() );

 fixedImageReader->Update();
 FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();
 resampler->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
 resampler->SetOutputOrigin( fixedImage->GetOrigin() );
 resampler->SetOutputSpacing( fixedImage->GetSpacing() );
 resampler->SetDefaultPixelValue( 0 );
 resampler->SetInterpolator( interpolator ); 
 
 //resampler->SetNumberOfThreads( 1 );
 std::cout<<"Number of threads used by the resampler: "<<resampler->GetNumberOfThreads()<<std::endl;

 // filter to cast the resampled to the fixed image
 typedef itk::CastImageFilter< FixedImageType, OutputImageType > CastFilterType;
 CastFilterType::Pointer caster = CastFilterType::New();
 
 // triger the pipeline
 caster->SetInput( resampler->GetOutput() );
 //writer->SetInput( caster->GetOutput() );

 OutputImageType::Pointer outputImage = OutputImageType::New();
 
 outputImage = caster->GetOutput();
 caster->Update();

 double myOrigin[] = {0, 0, 0}; // used to reset the origin of the DRR

 outputImage->Update();
 outputImage->SetOrigin( myOrigin ); 
 writer->SetInput( outputImage );
 
  try 
 { 
   std::cout << "Writing output image..." << std::endl;
   writer->Update();
 } 
 catch( itk::ExceptionObject & err ) 
 {      
   std::cerr << "ERROR: ExceptionObject caught !" << std::endl; 
   std::cerr << err << std::endl; 
 } 
 
 return 0;
 
}
