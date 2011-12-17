/*=========================================================================

 This program is a modified version of the typical itk registration pipeline.
 It is used to perform a 2D - 3D registration between the MR breast volume
 and the X-ray image.

=========================================================================*/

#include "LogHelper.h"
#include "ConversionUtils.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkImageRegistrationMethod.h"
#include "itkInvRayCastInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkInvResampleImageFilter.h"
#include "itkInvNormalizedCorrelationImageToImageMetric.h" 
#include "itkRegularStepGradientDescentOptimizer.h"

#include "itkAffineTransform2D3D.h"
#include "itkPCADeformationModelTransform.h"
#include "itkTranslationPCADeformationModelTransform.h"
#include "itkEuler3DTransform.h"
#include "itkCastImageFilter.h"
#include "itkTransformFactory.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"

#include "itkDataObject.h"
#include "itkDataObjectDecorator.h"
#include "itkCommand.h"
#include "itkNormalVariateGenerator.h"

#include "itkVector.h"
#include "itkVectorResampleImageFilter.h"
#include "itkVectorLinearInterpolateImageFunction.h"

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
    itkTypeMacro(FiniteDifferenceNCCImageToImageMetric, InvNormalizedCorrelationImageToImageMetric);
 
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
  typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
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
  if( argc < 11 ) //16 
 { 
   std::cerr << "Usage: " << std::endl;
   std::cerr << argv[0] << " fixedImageFile  movingImageFile  outputImageFile outputParametersFile xstart ystart xsize ysize PCAParametersDimension PCA_C0 PCA_C1 PCA_C2 PCA_C3 PCA_C4 PCA_C5";
   std::cerr <<  std::endl;
   return EXIT_FAILURE;
 }

 // values used for the DRR generation (see itkDRR).
 float sid = 660.;
 
 // input and output decl  
 const int dimension = 3;
 typedef float PixelType;  
 typedef float VolumePixelType; 
 typedef itk::Vector< double > VectorPixelType; // was:float

 typedef itk::Image< VectorPixelType,  dimension >   VectorImageType;

 typedef itk::Image< PixelType, dimension > FixedImageType;
 typedef itk::Image< VolumePixelType, dimension > MovingImageType; 
 typedef itk::Image< PixelType, dimension > OutputImageType; 
 
 // Reader and writer for the input and output images
 typedef itk::ImageFileReader< FixedImageType >  FixedReaderType;
 typedef itk::ImageFileReader< MovingImageType >  MovingReaderType;
 typedef itk::ImageFileReader< VectorImageType >  VectorReaderType;

 typedef itk::ImageFileWriter< OutputImageType >  WriterType;
 typedef itk::ImageFileWriter< VectorImageType >  VectorWriterType;

 FixedReaderType::Pointer  fixedImageReader = FixedReaderType::New();
 MovingReaderType::Pointer movingImageReader = MovingReaderType::New();
 WriterType::Pointer writer = WriterType::New();

 fixedImageReader->SetFileName( argv[1] );
 movingImageReader->SetFileName( argv[2] );
 writer->SetFileName( argv[3] );

 // Tranformation
 typedef itk::TranslationPCADeformationModelTransform< double, dimension > TransformType;

 ///////****************************************  
 // Prepare resampling of deformation field
  
 typedef itk::VectorResampleImageFilter< VectorImageType, VectorImageType, double > VectorResampleFilterType;

 VectorResampleFilterType::Pointer fieldResample = VectorResampleFilterType::New();

 VectorPixelType zeroDisplacement;
 zeroDisplacement[0] = 0.0;
 zeroDisplacement[1] = 0.0;
 zeroDisplacement[2] = 0.0;

 typedef VectorImageType::IndexType     VectorIndexType;
 typedef VectorImageType::RegionType    VectorRegionType;
 typedef VectorImageType::SizeType      VectorSizeType;
 typedef VectorImageType::SpacingType   VectorSpacingType;
 typedef VectorImageType::PointType     VectorPointType;
 typedef VectorImageType::DirectionType VectorDirectionType;

 VectorRegionType region;
 VectorSizeType size;
 VectorPointType origin;
 VectorSpacingType spacing;
 VectorDirectionType direction;
 VectorSizeType sizeNew;
 VectorSpacingType spacingNew;

 typedef itk::Euler3DTransform< double > RigidTransformType;
 RigidTransformType::Pointer rigidIdentityTransform = RigidTransformType::New();
 rigidIdentityTransform->SetIdentity();

 // Create the SDM transformation 
 int PCAParametersDimension = atoi( argv[9] ); 
 bool doResampleField = 1;// TO DO: CHANGE THAT, input from command line
 int factor = 2; // TO DO: CHANGE THAT, -//-
 std::stringstream sstr;

 char **filePCAcomponents = 0; 
 filePCAcomponents = &argv[10]; 
 
 typedef itk::ImageFileWriter < VectorImageType >  FieldWriterType;
 FieldWriterType::Pointer fieldWriter = FieldWriterType::New();

 itk::TransformFactoryBase::Pointer pTransformFactory = itk::TransformFactoryBase::GetFactory();
 TransformType::Pointer SDMTransform  = TransformType::New( );
 SDMTransform->SetNumberOfComponents(PCAParametersDimension);

 pTransformFactory->RegisterTransform(SDMTransform->GetTransformTypeAsString().c_str(),
	               		       SDMTransform->GetTransformTypeAsString().c_str(),
		             	       SDMTransform->GetTransformTypeAsString().c_str(), 1,
				       itk::CreateObjectFunction<TransformType>::New());
 typedef VectorImageType::Pointer    FieldPointer;
 typedef std::vector<FieldPointer>        FieldPointerArray;

 FieldPointerArray  fields(PCAParametersDimension+1);
 VectorReaderType::Pointer vectorImageReader = VectorReaderType::New();
                                          
 VectorImageType::Pointer sfield = VectorImageType::New();

 VectorWriterType::Pointer vectorImageWriter = VectorWriterType::New();
      
 sstr.str("");
 for (int k = 0; k <= PCAParametersDimension; k++ )
 {
   // read PCA displacement fields
   fields[k] = VectorImageType::New();
        
   niftk::LogHelper::InfoMessage(std::string("Loading component ") + filePCAcomponents[k]);
   vectorImageReader->SetFileName( filePCAcomponents[k] );

   try
   {
     vectorImageReader->Update();
   }
   catch( itk::ExceptionObject & excp )
   {
     std::cerr << excp << std::endl;
     return EXIT_FAILURE;
   }
   fields[k] = vectorImageReader->GetOutput();
   vectorImageReader->Update();

   niftk::LogHelper::InfoMessage("done");         

   if ((k==0) && (doResampleField))
   {
     // do resampling to uniform spacing as requested by user
     // assumes all fields are of the same format
                  
     niftk::LogHelper::InfoMessage("Change displacement fields spacing as requested");
     region = fields[k]->GetLargestPossibleRegion();
     size = fields[k]->GetLargestPossibleRegion().GetSize();
     origin = fields[k]->GetOrigin();
     spacing = fields[k]->GetSpacing();
     direction = fields[k]->GetDirection();

     for (int i = 0; i < dimension; i++ )
     {
       spacingNew[i] = factor;
       sizeNew[i] = (long unsigned int) niftk::Round(size[i]*spacing[i]/factor);
       niftk::LogHelper::InfoMessage("dim [" + niftk::ConvertToString( (int) i) + 
					    "] new spacing " + niftk::ConvertToString( spacingNew[i] ) + 
					    ", new size " + niftk::ConvertToString( sizeNew[i] ));
     }
                  
     fieldResample->SetSize( sizeNew );
     fieldResample->SetOutputSpacing( spacingNew );
     fieldResample->SetOutputOrigin(  origin );
     fieldResample->SetOutputDirection( direction );
     fieldResample->SetDefaultPixelValue( zeroDisplacement );
   }

   if (doResampleField)
   {
     // resample if necessary
     fieldResample->SetTransform( rigidIdentityTransform );
     fieldResample->SetInput(fields[k]);
     fieldResample->Update();
     fields[k] = fieldResample->GetOutput();
     fieldResample->Update();

     std::string filestem;
     std::string filename(filePCAcomponents[k]);
     std::string::size_type idx = filename.find_last_of('.');
     if (idx > 0)
       filestem = filename.substr(0, idx);
     else
       filestem = filePCAcomponents[k];

     sstr << filestem << "_Resampled" << ".mha";
     niftk::LogHelper::InfoMessage("Writing resampled component " + niftk::ConvertToString( (int) k ) + ": " + sstr.str());              

     fieldWriter->SetFileName( sstr.str() );
     sstr.str("");
     fieldWriter->SetInput( fields[k]);          
     try
     {
       fieldWriter->Update();
     }
     catch( itk::ExceptionObject & excp )
     {
       std::cerr << "Exception thrown " << std::endl;
       std::cerr << excp << std::endl;
     }
     niftk::LogHelper::InfoMessage("done");         
   }
          
   SDMTransform->SetFieldArray(k, fields[k]);
          
   fields[k]->DisconnectPipeline();
 }

 SDMTransform->Initialize();

 std::cout << "The SDM Transform:" << std::endl;
 SDMTransform->Print(std::cout);
 ///////****************************************


 // Optimizer 
 typedef itk::RegularStepGradientDescentOptimizer OptimizerType; 
 OptimizerType::Pointer optimizer = OptimizerType::New();
 
 // Metric type
 typedef itk::FiniteDifferenceNCCImageToImageMetric< FixedImageType, MovingImageType > MetricType;
 MetricType::Pointer metric = MetricType::New();
 
 // Interpolator type to evaluate intensities at non-grid positions
 typedef itk::InvRayCastInterpolateImageFunction< MovingImageType, double > InterpolatorType;
 InterpolatorType::Pointer interpolator = InterpolatorType::New();

 // Registration method
 typedef itk::ImageRegistrationMethod< FixedImageType, MovingImageType > RegistrationType;
 RegistrationType::Pointer registration = RegistrationType::New();

 // Parameters of the registration
 registration->SetTransform( SDMTransform ); //regTransform );
 registration->SetOptimizer( optimizer );
 registration->SetMetric( metric );
 registration->SetInterpolator( interpolator );
 registration->SetFixedImage( fixedImageReader->GetOutput() );
 registration->SetMovingImage( movingImageReader->GetOutput() );

 // Update the fixedImage to obtain the buffered area that will be used as the
 // region for the metric to be computed 
 FixedImageType::SizeType fixedSize;
 fixedSize[0] = atoi( argv[7] );
 fixedSize[1] = atoi( argv[8] );
 fixedSize[2] = 1;

 // set the region for the input and the output images 
 FixedImageType::RegionType inputRegion;
 inputRegion.SetSize( fixedSize );
 FixedImageType::RegionType::IndexType inputStart;
 inputStart[0] = atoi( argv[5] );
 inputStart[1] = atoi( argv[6] );
 inputStart[2] = 0;
 inputRegion.SetIndex( inputStart );

 fixedImageReader->Update();
 registration->SetFixedImageRegion ( inputRegion );

 // set the origin for the 3D volume ?? -> TO DO: Do I need that?
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

 // set the origin for the 2D image
 double origin2D[ dimension ];

 const itk::Vector<double, 3> resolution2D = fixedImageReader->GetOutput()->GetSpacing();
 typedef FixedImageType::RegionType      ImageRegionType2D;
 typedef ImageRegionType2D::SizeType  SizeType2D;

 ImageRegionType2D region2D = fixedImageReader->GetOutput()->GetBufferedRegion();
 SizeType2D        size2D   = region2D.GetSize();

 origin2D[0] = - resolution2D[0]*((double) size2D[0] - 1.)/2.; 
 origin2D[1] = - resolution2D[1]*((double) size2D[0] - 1.)/2.;  
 origin2D[2] = + 90. ; //100
 
 fixedImageReader->GetOutput()->SetOrigin( origin2D );

 std::cout <<"2D origin: "<<"["<<origin2D[0]<<" ,"<<origin2D[1]<<" ,"<<origin2D[2]<<"]"<< std::endl;

 // Initialisation of the interpolator
 InterpolatorType::InputPointType focalpoint;

 focalpoint[0]= 0;
 focalpoint[1]= 0;
 focalpoint[2]= - (sid-90.); //100

 std::cout << "Focal point: " << focalpoint << std::endl;

 interpolator->SetFocalPoint( focalpoint );
 interpolator->SetTransform( SDMTransform );

 // Initialise the registration
 registration->SetInitialTransformParameters( SDMTransform->GetParameters() );

 //subtract the mean to create more steap valeys
 metric->SetSubtractMean(true);
 
 optimizer->MaximizeOff();
 optimizer->SetMaximumStepLength( 0.50 ); //1.00
 optimizer->SetMinimumStepLength( 0.01 ); //0.001;
 optimizer->SetNumberOfIterations( 50 );//10 );//100 );//400 );
 optimizer->SetRelaxationFactor( 0.8 );
 
 // Optimizer weightings - > TO DO : r they needed? Fill with ones?
 itk::Optimizer::ScalesType weightings( SDMTransform->GetNumberOfParameters() );

 for (int i=0; i<PCAParametersDimension; i++)
 {
   weightings[i] = 10.0;
   std::cout << "Weight " << i << " = " << weightings[i] << std::endl;
 }
 for (unsigned int i=PCAParametersDimension; i<SDMTransform->GetNumberOfParameters(); i++)
 {
   weightings[i] = 1.0;
   std::cout << "Weight " << i << " = " << weightings[i] << std::endl;
 }

 optimizer->SetScales( weightings );

 myFile.open( argv[4] );

 myFile << "0 = 0 :";
 myFile <<  SDMTransform->GetParameters()  << std::endl;

 std::cout << "Initial Parameters" << " : "; 
 std::cout <<  SDMTransform->GetParameters()  << std::endl;

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
