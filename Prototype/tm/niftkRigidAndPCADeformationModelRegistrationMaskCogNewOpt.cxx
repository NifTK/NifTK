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
//#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkConstrainedRegStepOptimizerForSDM.h"

#include "itkAffineTransform2D3D.h"
#include "itkPCADeformationModelTransform.h"
#include "itkRigidPCADeformationModelTransform.h"
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

#include "itkNormalVariateGenerator.h"
#include "itkImageMomentsCalculator.h"

#include "itkImageMaskSpatialObject.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <math.h>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main( int argc, char * argv[] )
{
  if( argc < 11 ) //16 
 { 
   std::cerr << "Usage: " << std::endl;
   std::cerr << argv[0] << " fixedImageFile  movingImageFile outputImageFile outputParametersFile fixedImageMaskFile numberOfIterations PCAParametersDimension initialRolling initialInPlane PCA_C0 PCA_C1 PCA_C2 PCA_C3 PCA_C4 PCA_C5";
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
 typedef itk::RigidPCADeformationModelTransform< double, dimension > TransformType;

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
 int PCAParametersDimension = atoi( argv[7] ); 
 bool doResampleField = 0;//1;// TO DO: CHANGE THAT, input from command line
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
 
 // Set the centre for the rotations of the rigid transformation
 // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 typedef itk::ImageMomentsCalculator< MovingImageType >  ImageCalculatorType;
 ImageCalculatorType::Pointer imageCalculator = ImageCalculatorType::New();

 movingImageReader->Update();
 imageCalculator->SetImage(movingImageReader->GetOutput());
 imageCalculator->Compute();
 ImageCalculatorType::VectorType massCentreMoving = imageCalculator->GetCenterOfGravity();

 std::cout<<"Mass centre of the MOVING image: "<<massCentreMoving[0]<<" "<<massCentreMoving[1]<<" "<<massCentreMoving[2]<<std::endl;
  
 TransformType::InputPointType transformCentre; 
 transformCentre[0] = massCentreMoving[0];
 transformCentre[1] = massCentreMoving[1];
 transformCentre[2] = massCentreMoving[2];
    
 SDMTransform->SetCentre( transformCentre );

 std::cout << "The SDM Transform:" << std::endl;
 SDMTransform->Print(std::cout);
 ///////****************************************


 // Optimizer 
 //typedef itk::RegularStepGradientDescentOptimizer OptimizerType; 
 typedef itk::ConstrainedRegStepOptimizerForSDM OptimizerType; 
 OptimizerType::Pointer optimizer = OptimizerType::New();
 
 // Metric type
 typedef itk::InvNormalizedCorrelationImageToImageMetric< FixedImageType, MovingImageType > MetricType;
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

 // Used because of the ray-casting
 double halfDim3D[ dimension ];

 const itk::Vector<double, 3> resolution3D = movingImageReader->GetOutput()->GetSpacing();

 typedef MovingImageType::RegionType  ImageRegionType3D;
 typedef ImageRegionType3D::SizeType  SizeType3D;

 ImageRegionType3D region3D = movingImageReader->GetOutput()->GetBufferedRegion();
 SizeType3D        size3D   = region3D.GetSize();

 halfDim3D[0] = resolution3D[0]*((double) size3D[0]-1)/2.; 
 halfDim3D[1] = resolution3D[1]*((double) size3D[1]-1)/2.; 
 halfDim3D[2] = resolution3D[2]*((double) size3D[2]-1)/2.;

 movingImageReader->Update();
 
 // set the origin for the 2D image
 double origin2D[ dimension ];

 typedef itk::ImageMomentsCalculator< FixedImageType >  FixedImageCalculatorType;
 FixedImageCalculatorType::Pointer fixedImageCalculator = FixedImageCalculatorType::New();
 fixedImageReader->Update();
 fixedImageCalculator->SetImage(fixedImageReader->GetOutput());
 fixedImageCalculator->Compute();
 FixedImageCalculatorType::VectorType massCentreFixed = fixedImageCalculator->GetCenterOfGravity();

 std::cout<<"Mass centre of the FIXED image: "<<massCentreFixed[0]<<" "<<massCentreFixed[1]<<std::endl;

 origin2D[0] = massCentreMoving[0] - halfDim3D[0] - massCentreFixed[0]; 
 origin2D[1] = massCentreMoving[1] - halfDim3D[1] - massCentreFixed[1];  
 origin2D[2] = massCentreMoving[2] - halfDim3D[2] + 90. ; //90
 
 fixedImageReader->GetOutput()->SetOrigin( origin2D );

 std::cout <<"2D origin: "<<"["<<origin2D[0]<<" ,"<<origin2D[1]<<" ,"<<origin2D[2]<<"]"<< std::endl;

 ////////// Set the mask to the metric///////////////
 typedef itk::ImageMaskSpatialObject< dimension >   MaskType;
 MaskType::Pointer  spatialObjectMask = MaskType::New();
 
 typedef itk::Image< unsigned char, dimension >   ImageMaskType;
 typedef itk::ImageFileReader< ImageMaskType >    MaskReaderType;
 MaskReaderType::Pointer  maskReader = MaskReaderType::New();
 ImageMaskType::Pointer maskImage = ImageMaskType::New();

 maskReader->SetFileName( argv[5] );

 maskReader->Update(); 
 maskImage =  maskReader->GetOutput();
 maskImage->SetOrigin( origin2D );
 spatialObjectMask->SetImage( maskImage );

 metric->SetFixedImageMask( spatialObjectMask );

 ////////////////////////////////////////


 // Initialisation of the interpolator
 InterpolatorType::InputPointType focalpoint;

 focalpoint[0] = massCentreMoving[0] - halfDim3D[0];
 focalpoint[1] = massCentreMoving[1] - halfDim3D[1];
 focalpoint[2] = massCentreMoving[2] - halfDim3D[2] - (sid-90.); //90

 std::cout << "Focal point: " << focalpoint << std::endl;

 interpolator->SetFocalPoint( focalpoint );
 interpolator->SetTransform( SDMTransform );

 //Use SetParameters to the transform for initial rotations
 //use the test image with the spike to test initial position
 typedef  TransformType::ParametersType ParametersType;
 ParametersType initialParameters (SDMTransform->GetNumberOfParameters()); 

 initialParameters = SDMTransform->GetParameters();

 // CHANGE THESE NOT TO BE HARD CODED
 // Rolling
 initialParameters[PCAParametersDimension] = M_PI/180.0*( atof(argv[8]) ); 
 // In-plane rotation
 initialParameters[PCAParametersDimension+1] = -M_PI/180.0*( atof(argv[9]) );

 SDMTransform->SetParameters( initialParameters );

 // Initialise the registration
 registration->SetInitialTransformParameters( initialParameters );

 //subtract the mean to create more steap valeys
 metric->SetSubtractMean(true);
 
 optimizer->MaximizeOff();
 optimizer->SetMaximumStepLength( 1.0 ); //1.00 //0.50
 optimizer->SetMinimumStepLength( 0.1 ); //0.01;
 optimizer->SetNumberOfIterations( atoi( argv[6] ) );
 optimizer->SetRelaxationFactor( 0.8 );
 
 // Optimizer weightings 
 itk::Optimizer::ScalesType weightings( SDMTransform->GetNumberOfParameters() );

 optimizer->SetProgressFileName( argv[4] );

 for (int i=0; i<PCAParametersDimension; i++) // pca components
 {
   weightings[i] = 0.5;
   std::cout << "Weight " << i << " = " << weightings[i] << std::endl;
 }
 for (int i=PCAParametersDimension; i<PCAParametersDimension+2; i++) // rotations
 {
   weightings[i] = 100.0;
   std::cout << "Weight " << i << " = " << weightings[i] << std::endl;
 }
 for (unsigned int i=PCAParametersDimension+2; i<SDMTransform->GetNumberOfParameters(); i++) //translations
 {
   weightings[i] = 0.1;//1.0;
   std::cout << "Weight " << i << " = " << weightings[i] << std::endl;
 }

 optimizer->SetScales( weightings );
 optimizer->SetNumberOfPCAComponents( PCAParametersDimension );
 

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
