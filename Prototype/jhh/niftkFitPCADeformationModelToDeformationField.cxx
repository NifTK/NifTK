
/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: jhh                 $
 $Date:: 2011-12-16 15:12:34 +#$
 $Rev:: 8054                   $

 Copyright (c) UCL : See the file LICENSE.txt in the top level
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkImageRegistrationFactory.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkAddImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkMultiplyImageFilter.h"

#include "itkImage.h"
#include "itkImageRegionIterator.h"

#include "itkAffineTransform.h"
#include "itkTransformFileWriter.h"
#include "itkEuler3DTransform.h"
#include "itkVectorResampleImageFilter.h"

#include "itkKnownCorEuclideanDistancePointMetric.h"
#include "itkLevenbergMarquardtOptimizer.h"
#include "itkPointSet.h"
#include "itkPointSetToPointSetRegistrationMethod.h"


struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_SWITCH, "dbg", 0, "Output debugging information."},
  {OPT_SWITCH, "v", 0,   "Verbose output during execution."},

  {OPT_SWITCH, "trans", 0,       "Optimise translation as well as PCA components."},
  {OPT_SWITCH, "forceCentre", 0, "Use if PCAFiles are centred but not images."},

  {OPT_INT,    "maxIters",       "value",   "Maximum number of iterations [default 300]."},
  {OPT_FLOAT,  "resample",       "spacing", "Resample PCA displacement fields to have uniform spacing."},

  {OPT_INT, "mvalue", "value", "The mask intensity used to determine the region of interest."},
  {OPT_STRING, "mask", "filename", "Calculate fit and error only over mask region."},

  {OPT_STRING, "oi", "filename", "Output an image of the error at each voxel."},

  {OPT_STRING, "dofout",  "filename", "Output transformation."},
  {OPT_STRING, "defout",  "filename", "Output deformation field."},

  {OPT_STRING|OPT_REQ, "defin", "filename", "The target input deformation field."},

  {OPT_STRING, "tAffinePCA", "filename", "Apply an affine transformation to the PCA def'n model to transform to the fixed image space."},
  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "Input PCA eigen deformations (one '.mha' vector file per component)."},
  {OPT_MORE, NULL, "...", NULL},

  {OPT_DONE, NULL, NULL, 
   "A program to fit a PCA deformation model to a known deformation field. Outputs the resulting error. "
   "The motivation for this program was to determine the relative merits of sub-sampling the deformation fields "
   "versus varying the number of PCA components."
  }
};


enum {
  O_DEBUG = 0,
  O_VERBOSE,

  O_TRANSLATION,
  O_FORCE_CENTRE,

  O_MAX_NUMBER_OF_ITERATIONS,
  O_RESAMPLE,

  O_MASK_VALUE,
  O_MASK,

  O_OUTPUT_IMAGE,

  O_OUTPUT_TRANSFORMATION,
  O_OUTPUT_DEFORMATION,
  
  O_TARGET_DEFORMATION,

  O_PCA_AFFINE,
  O_PCA_EIGEN_DEFORMATIONS,
  O_MORE
};

  
// CommandIterationUpdate
class CommandIterationUpdate : public itk::Command 
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef itk::SmartPointer<Self>   Pointer;
  itkNewMacro( Self );

protected:
  CommandIterationUpdate() {};

public:

  typedef itk::LevenbergMarquardtOptimizer     OptimizerType;
  typedef const OptimizerType *                OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event)
    {
    Execute( (const itk::Object *)caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event)
    {
    OptimizerPointer optimizer = 
                         dynamic_cast< OptimizerPointer >( object );

    if( ! itk::IterationEvent().CheckEvent( &event ) )
      {
      return;
      }

    //std::cout << "Value = " << optimizer->GetCachedValue() << std::endl; 
    std::cout << "Position = "  << optimizer->GetCachedCurrentPosition();
    std::cout << std::endl << std::endl;

    }
   
};


// Global declarations
typedef double VectorComponentType;    // Eigen vector displacement type
const unsigned int ImageDimension = 3; // 3D images


// -------------------------------------------------------------------
// This is effectively the 'main' function but we've templated it over
// the transform type.
// -------------------------------------------------------------------

template<class TransformType>
int FitPCADeformationModelToDeformationField(int argc, char** argv)
{
  bool debug;                    // Output debugging information
  bool verbose;                  // Verbose output during execution
  bool doResampleSDM = false;
  bool doResampleField = false;
  bool flgForceCentre = false;

  // Default filenames
  char *fileInputDeformationField = NULL;

  char *maskName = NULL;
  char *outName = NULL;

  char *filePCAaffine = 0;   
  char *filePCAcomponent = 0;   
  char **filePCAcomponents = 0; 

  unsigned int i;		        // Loop counter
  unsigned int nPCAcomponents = 0;	// The number of input PCA components

  unsigned int PCAParametersDimension = 0;   

  unsigned int maxNumberOfIterations = 300;

  int arg;			// Index of arguments in command line 

  float factor = 1.0;

  std::string filePCAdeformations;
  std::string fileOutputTransformation;
  std::string fileOutputDeformation;

  std::stringstream sstr;

  typedef int MaskPixelType;
  MaskPixelType mask_value = 1;
  
  // Image and reader definitions
  
  typedef float InternalPixelType;
  typedef itk::Vector< VectorComponentType, ImageDimension > VectorPixelType;
  typedef itk::Image< VectorPixelType,  ImageDimension >   DeformationFieldType;

  typedef DeformationFieldType::Pointer FieldPointer;
  typedef std::vector<FieldPointer>     FieldPointerArray;

  typedef itk::ImageRegionIteratorWithIndex<DeformationFieldType>  FieldIterator;

  typedef DeformationFieldType::PixelType     DisplacementType;  
  typedef DeformationFieldType::IndexType     FieldIndexType;
  typedef DeformationFieldType::RegionType    FieldRegionType;
  typedef DeformationFieldType::SizeType      FieldSizeType;
  typedef DeformationFieldType::SpacingType   FieldSpacingType;
  typedef DeformationFieldType::PointType     FieldPointType;
  typedef DeformationFieldType::DirectionType FieldDirectionType;
  
  typedef itk::ImageFileReader < DeformationFieldType >  FieldReaderType;
  typedef itk::VectorResampleImageFilter< DeformationFieldType, DeformationFieldType > FieldResampleFilterType;

  typedef itk::Image< MaskPixelType,  ImageDimension >    MaskImageType;
  typedef itk::ImageRegionIterator<MaskImageType>    MaskIterator;

  typedef itk::ImageFileReader< MaskImageType  >  MaskReaderType;

  typedef itk::Image< InternalPixelType,  ImageDimension >    InternalImageType;
  
  typedef itk::ImageFileWriter< InternalImageType  >  InternalImageWriterType;

  typedef itk::ImageRegionIterator<InternalImageType>  ImageIterator;


  // Declare the type for filters
  typedef itk::AddImageFilter<
  InternalImageType,
    InternalImageType,
    InternalImageType  >       addFilterType;
  
  typedef itk::SubtractImageFilter<
  InternalImageType,
    InternalImageType,
    InternalImageType  >       subFilterType;
  
  typedef itk::MultiplyImageFilter<
  InternalImageType,
    InternalImageType,
    InternalImageType > multiplyFilterType;


  typedef addFilterType::Pointer                addFilterTypePointer;
  typedef subFilterType::Pointer                subFilterTypePointer;
  typedef multiplyFilterType::Pointer           multiplyFilterTypePointer;

  // Parse the command line
  // ~~~~~~~~~~~~~~~~~~~~~~
  
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_FORCE_CENTRE, flgForceCentre);

  CommandLineOptions.GetArgument(O_MAX_NUMBER_OF_ITERATIONS, maxNumberOfIterations);

  if (CommandLineOptions.GetArgument(O_RESAMPLE, factor))
    doResampleSDM = true;

  CommandLineOptions.GetArgument(O_MASK_VALUE, mask_value);
  CommandLineOptions.GetArgument(O_MASK, maskName);

  CommandLineOptions.GetArgument(O_OUTPUT_IMAGE, outName);

  CommandLineOptions.GetArgument(O_TARGET_DEFORMATION, fileInputDeformationField);

  CommandLineOptions.GetArgument(O_OUTPUT_TRANSFORMATION, fileOutputTransformation);
  CommandLineOptions.GetArgument(O_OUTPUT_DEFORMATION, fileOutputDeformation);

  // Get the PCA component filenames

  CommandLineOptions.GetArgument(O_PCA_AFFINE, filePCAaffine);

  CommandLineOptions.GetArgument(O_PCA_EIGEN_DEFORMATIONS, filePCAdeformations);
  CommandLineOptions.GetArgument(O_MORE, arg);
  
  if (arg < argc) {		   // Many deformation fields
    nPCAcomponents = argc - arg;
    filePCAcomponents = &argv[arg-1];

    std::cout << "SDM deformation fields: ";
    for (i=0; i<=nPCAcomponents; i++)
      std::cout <<  niftk::ConvertToString( (int) i+1) + " " + filePCAcomponents[i];
  }
  else if (filePCAcomponent) { // Single deformation field
    nPCAcomponents = 1;
    filePCAcomponents = &filePCAcomponent;

    std::cout << "SDM deformation field: ") + filePCAcomponents[0];
  }
  else {
    nPCAcomponents = 0;
    filePCAcomponents = 0;
  }

  PCAParametersDimension = nPCAcomponents;   


  // Create the image readers etc.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  FieldReaderType::Pointer fieldReader1 = FieldReaderType::New();
                                           
  DeformationFieldType::Pointer targetField;
  DeformationFieldType::Pointer sdmField;

  MaskReaderType::Pointer maskReader = MaskReaderType::New();

  MaskImageType::Pointer mask;

  InternalImageWriterType::Pointer writer = InternalImageWriterType::New();

  InternalImageType::Pointer errorImage;
  InternalImageType::Pointer initialerrorImage;
    
  addFilterTypePointer addfilter = addFilterType::New();
  subFilterTypePointer subfilter = subFilterType::New();
  multiplyFilterTypePointer multiplyfilter = multiplyFilterType::New();


  // Prepare resampling of deformation field if needed
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  FieldResampleFilterType::Pointer fieldResample = FieldResampleFilterType::New();

  VectorPixelType zeroDisplacement;
  zeroDisplacement[0] = 0.0;
  zeroDisplacement[1] = 0.0;
  zeroDisplacement[2] = 0.0;

  typedef itk::Euler3DTransform< double > RigidTransformType;
  RigidTransformType::Pointer rigidIdentityTransform = RigidTransformType::New();
  rigidIdentityTransform->SetIdentity();
  
  typedef DeformationFieldType::IndexType     FieldIndexType;
  typedef DeformationFieldType::RegionType    FieldRegionType;
  typedef DeformationFieldType::SizeType      FieldSizeType;
  typedef DeformationFieldType::SpacingType   FieldSpacingType;
  typedef DeformationFieldType::PointType     FieldPointType;
  typedef DeformationFieldType::DirectionType FieldDirectionType;

  FieldSizeType sizeNew;
  FieldSpacingType spacingNew;



  // Read the deformation field
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  fieldReader1->SetFileName(fileInputDeformationField);

  std::cout << "Reading deformation field: " << std::string(fileInputDeformationField);

  try
    {
      fieldReader1->Update();
    }

  catch( itk::ExceptionObject & excp )
    {
      std::cerr << "Exception thrown " << std::endl;
      std::cerr << excp << std::endl;
    }

  targetField = fieldReader1->GetOutput();

  if (debug)
    targetField->Print(std::cout); 
  
  FieldRegionType region = targetField->GetLargestPossibleRegion();
  FieldSizeType size = targetField->GetLargestPossibleRegion().GetSize();
  FieldPointType origin = targetField->GetOrigin();
  FieldSpacingType spacing = targetField->GetSpacing();
  FieldDirectionType direction = targetField->GetDirection();
  

  // Read the mask image
  // ~~~~~~~~~~~~~~~~~~~

  if (maskName != NULL) {

    maskReader->SetFileName(maskName);
    try
      {
	maskReader->Update();
      }
    catch( itk::ExceptionObject & excp )
      {
	std::cerr << "Exception thrown " << std::endl;
	std::cerr << excp << std::endl;
      }
    mask = maskReader->GetOutput();

    FieldRegionType regionM = mask->GetLargestPossibleRegion();
    FieldSizeType sizeM = mask->GetLargestPossibleRegion().GetSize();
    FieldPointType originM = mask->GetOrigin();
    FieldSpacingType spacingM = mask->GetSpacing();
    FieldDirectionType directionM = mask->GetDirection();

    for (unsigned int i = 0; i < ImageDimension; i++ ) {
      
      if (spacingM[i]!=spacing[i]) {
	doResampleField = true;
	std::cout << "Resampling field 1 relative to mask, spacing ["
				      << niftk::ConvertToString( (int) i ) << "] "
				      << niftk::ConvertToString( spacing[i] ) << " to "
				      << niftk::ConvertToString( spacingM[i] ) ;
      }

      if (sizeM[i]!=size[i]) {

	doResampleField = true;
	std::cout << "Resampling field 1 relative to mask, size ["
				      << niftk::ConvertToString( (int) i ) << "] "
				      << niftk::ConvertToString( size[i] ) << " to "
				      << niftk::ConvertToString( sizeM[i] ) ;
      }

      if (originM[i]!=origin[i]) {

	doResampleField = true;
	std::cout << "Resampling field 1 relative to mask, origin ["
				      << niftk::ConvertToString( (int) i ) << "] "
				      << niftk::ConvertToString( origin[i] ) << " to "
				      << niftk::ConvertToString( originM[i] );
      }

      for (unsigned int j = 0; j < ImageDimension; j++ ) {

	if (directionM[i][j]!=direction[i][j]) {
	  doResampleField = true;
	  std::cout << "Resampling field 1 relative to mask, direction ["
					<< niftk::ConvertToString( (int) i ) << "]["
					<< niftk::ConvertToString( (int) j ) << "] "
					<< niftk::ConvertToString( direction[i][j] ) << " to "
					<< niftk::ConvertToString( directionM[i][j] );
	}
      }
    }

    if (doResampleField) {

      std::cout << "Changing field 1 to image format size and spacing of mask";

      fieldResample->SetSize( sizeM );
      fieldResample->SetOutputOrigin( originM );
      fieldResample->SetOutputSpacing( spacingM );
      fieldResample->SetOutputDirection( directionM );
      fieldResample->SetDefaultPixelValue( zeroDisplacement );
      fieldResample->SetTransform( rigidIdentityTransform );
                  
      fieldResample->SetInput(targetField);
      fieldResample->Update();

      targetField = fieldResample->GetOutput();
      targetField->DisconnectPipeline();

      region = regionM;
      size = sizeM;
      origin = originM;
      spacing = spacingM;
      direction = directionM;

      std::cout << "Deformation field: " << std::endl;
      targetField->Print(std::cout);
    }
  }
  

  // Read in affine transformation to transform SDM to fixed/target image space if required
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::AffineTransform<double,Dimension> AffineTransformType;
  AffineTransformType::Pointer affineTransform = AffineTransformType::New();
  affineTransform->SetIdentity();

  AffineTransformType::Pointer affineTransformInverse = AffineTransformType::New();
  typedef AffineTransformType::ParametersType AffineParametersType;
  double affineDetOfJac = 1.0;

  if ( filePCAaffine != NULL ) {

    affineReader->SetFileName( filePCAaffine);
    affineReader->Update();

    TransformListType* transforms = affineReader->GetTransformList();

    itk::TransformFileReader::TransformListType::const_iterator itA = transforms->begin();

    if(!strcmp((*itA)->GetNameOfClass(),"AffineTransform")) {

      affineTransform = static_cast<AffineTransformType*>((*itA).GetPointer());
	
      //affineTransform->Print(std::cout);
					
      affineTransform->GetInverse(affineTransformInverse);
					
      // determine volume change of affine transformation
					
      AffineParametersType affineParameters = affineTransformInverse->GetParameters();
					
      std::cout << "Par: " << affineParameters << std::endl;
      vnl_matrix<double> p(3, 3);
      p[0][0] = (double) affineParameters[0];
      p[0][1] = (double) affineParameters[1];
      p[0][2] = (double) affineParameters[2];
      p[1][0] = (double) affineParameters[3];
      p[1][1] = (double) affineParameters[4];
      p[1][2] = (double) affineParameters[5];
      p[2][0] = (double) affineParameters[6];
      p[2][1] = (double) affineParameters[7];
      p[2][2] = (double) affineParameters[8];
      
      double detA1 = p[0][0]*(p[1][1]*p[2][2] - p[1][2]*p[2][1]);
      double detA2 = p[0][1]*(p[1][0]*p[2][2] - p[1][2]*p[2][0]);
      double detA3 = p[0][2]*(p[1][0]*p[2][1] - p[1][1]*p[2][0]);
      affineDetOfJac = detA1-detA2+detA3;
      std::cout << " det(A) = " << affineDetOfJac << std::endl;
    }
  }
	

  // Create the SDM transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  itk::TransformFactoryBase::Pointer pTransformFactory = itk::TransformFactoryBase::GetFactory();
  typename TransformType::Pointer SDMTransform  = TransformType::New( );
  SDMTransform->SetNumberOfComponents(PCAParametersDimension);

  pTransformFactory->RegisterTransform(SDMTransform->GetTransformTypeAsString().c_str(),
				       SDMTransform->GetTransformTypeAsString().c_str(),
				       SDMTransform->GetTransformTypeAsString().c_str(),
				       1,
				       itk::CreateObjectFunction<TransformType>::New());

  FieldPointerArray  fields(PCAParametersDimension+1);
  FieldReaderType::Pointer fieldReader = FieldReaderType::New();
                                           
  DeformationFieldType::Pointer sfield = DeformationFieldType::New();

  typedef itk::ImageFileWriter < DeformationFieldType >  FieldWriterType;
  FieldWriterType::Pointer fieldWriter = FieldWriterType::New();
      
  sstr.str("");
  for (unsigned int k = 0; k <= PCAParametersDimension; k++ ) {

    // read PCA displacement fields
    fields[k] = DeformationFieldType::New();
          
      std::cout << "Loading component " + filePCAcomponents[k];
      fieldReader->SetFileName( filePCAcomponents[k] );

    try {
      fieldReader->Update();
    }
    catch( itk::ExceptionObject & excp ) {
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }

      std::cout << "done";

    niftk::LogHelper::InfoMessage("done");         

    if ((k==0) && (doResampleSDM)) {

      // do resampling to uniform spacing as requested by user
      // assumes all fields are of the same format
                  
	  std::cout << "Change displacement fields spacing as requested";

	  for (unsigned int i = 0; i < ImageDimension; i++ )
	    {
	      spacingNew[i] = factor;
        sizeNew[i] = (long unsigned int) niftk::Round(size[i]*spacing[i]/factor);
	      std::cout << "dim [" << niftk::ConvertToString( (int) i) <<
					    "] new spacing " << niftk::ConvertToString( spacingNew[i] ) <<
					    ", new size " << niftk::ConvertToString( sizeNew[i] );
	    }
	}
      }

      // Write resampled components out

      std::string filestem;
      std::string filename(filePCAcomponents[k]);
      std::string::size_type idx = filename.find_last_of('.');

      if (idx > 0)
	filestem = filename.substr(0, idx);
      else
	filestem = filePCAcomponents[k];
	
      sstr << filestem << "_Resampled" << ".mha";
      niftk::LogHelper::InfoMessage("Writing resampled component " + niftk::ConvertToString( (int) k ) + ": " + sstr.str());              

	  sstr << filestem << "_Resampled" << ".mha";
	  std::cout << "Writing resampled component " << niftk::ConvertToString( (int) k ) << ": " << sstr.str();

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
	  std::cout << "done";
	}
          
    SDMTransform->SetFieldArray(k, fields[k]);
      
    fields[k]->DisconnectPipeline();
  }

  SDMTransform->Initialize();

  if (debug) {
    std::cout << "The SDM Transform:" << std::endl;
    SDMTransform->Print(std::cout);
  }


  // Create the two point sets
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::PointSet< float, ImageDimension >   PointSetType;
  
  // This will be the set of image voxel coordinates
  PointSetType::Pointer fixedPointSet  = PointSetType::New(); 
  // This will be the set of coordinates displaced by the deformation field
  PointSetType::Pointer movingPointSet = PointSetType::New();


  typedef PointSetType::PointType     PointType;

  typedef PointSetType::PointsContainer  PointsContainer;

  PointsContainer::Pointer fixedPointContainer  = PointsContainer::New();
  PointsContainer::Pointer movingPointContainer = PointsContainer::New();

  PointType fixedPoint;
  PointType movingPoint;
  
  FieldIndexType Findex;
  DisplacementType displacement1;
  
  unsigned int pointID = 0;

  if (maskName != NULL) {

      Iterator itM( mask, mask->GetLargestPossibleRegion() );
      for ( itM.Begin(); !itM.IsAtEnd(); ++itM )
	{
	  value = itM.Get();
	  if (value == maskValue)
	    {
	      index = itM.GetIndex();                   
	      mask->TransformIndexToPhysicalPoint( index, fixedPoint );
	      field->TransformPhysicalPointToIndex( fixedPoint, indexF );
	      // nearest neighbour, need proper interpolation
	      displacementVector = field->GetPixel(indexF);
	      movingPoint[0] = fixedPoint[0] + displacementVector[0];
	      movingPoint[1] = fixedPoint[1] + displacementVector[1];
	      movingPoint[2] = fixedPoint[2] + displacementVector[2];
	      fixedLandmarks.push_back(fixedPoint);
	      movingLandmarks.push_back(movingPoint);
	      pointId++;
	      if (pointId < 5)
		printf("displacementVector %f %f %f\n",displacementVector[0],displacementVector[1],displacementVector[2]);
	    }
	}

  }
  else {

    FieldIterator itField1( targetField, targetField->GetLargestPossibleRegion() );

    for ( itField1.GoToBegin(); !itField1.IsAtEnd(); ++itField1 ) {

      Findex = itField1.GetIndex();
	  
      targetField->TransformIndexToPhysicalPoint( Findex, movingPoint );
      movingPointContainer->InsertElement(pointID, movingPoint);

      displacement1 = itField1.Get();              
      fixedPoint = movingPoint + displacement1;
      fixedPointContainer->InsertElement(pointID, fixedPoint);

      pointID++;

#if 0
      std::cout << pointID 
		<< " Index: " << Findex 
		<< " Fixed point: " << fixedPoint 
		<< " Displacement: " << displacement1 
		<< " Moving point: " << movingPoint << std::endl;
#endif
    }
  }

  fixedPointSet->SetPoints( fixedPointContainer );
  movingPointSet->SetPoints( movingPointContainer );


  std::cout << "Fixed point set: " << fixedPointSet << std::endl;
  std::cout << "Moving point set: " << movingPointSet << std::endl;
  std::cout << "Number of data points: " << pointID << std::endl;


  // Fit the SDM to the deformation field
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  // Set up  the Metric
  typedef itk::KnownCorEuclideanDistancePointMetric< PointSetType, PointSetType> MetricType;

  typedef MetricType::TransformType                 TransformBaseType;
  typedef TransformBaseType::ParametersType         ParametersType;
  typedef TransformBaseType::JacobianType           JacobianType;

  MetricType::Pointer  metric = MetricType::New();

  // Optimizer Type
  typedef itk::LevenbergMarquardtOptimizer OptimizerType;

  OptimizerType::Pointer optimizer = OptimizerType::New();
  optimizer->SetUseCostFunctionGradient(false);

  // Registration Method
  typedef itk::PointSetToPointSetRegistrationMethod< PointSetType, PointSetType > RegistrationType;


  RegistrationType::Pointer registration = RegistrationType::New();


  // Scale the translation components of the Transform in the Optimizer
  OptimizerType::ScalesType scales( SDMTransform->GetNumberOfParameters() );
  scales.Fill( 1.0 );
  
  double gradientTolerance  =  1e-5;    // convergence criterion
  double valueTolerance     =  1e-5;    // convergence criterion
  double epsilonFunction    =  1e-6;   // convergence criterion


  optimizer->SetScales( scales );

  optimizer->SetNumberOfIterations( maxNumberOfIterations );

  optimizer->SetValueTolerance( valueTolerance );
  optimizer->SetGradientTolerance( gradientTolerance );
  optimizer->SetEpsilonFunction( epsilonFunction );

  registration->SetInitialTransformParameters( SDMTransform->GetParameters() );

  // Connect all the components required for Registration
  registration->SetMetric( metric );
  registration->SetOptimizer( optimizer );
  registration->SetTransform( SDMTransform );
  registration->SetFixedPointSet( fixedPointSet );
  registration->SetMovingPointSet( movingPointSet );

  // Connect an observer
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );

  try 
    {
    registration->StartRegistration();
    }
  catch( itk::ExceptionObject & e )
    {
    std::cout << e << std::endl;
    return EXIT_FAILURE;
    }

  std::cout << "Solution = " << SDMTransform->GetParameters() << std::endl;


  // Write the transformation coefficients
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  SDMTransform = dynamic_cast<TransformType*>(registration->GetTransform());

  if ( fileOutputTransformation.length() > 0 ) {

    itk::TransformFileWriter::Pointer SDMWriter;
    SDMWriter = itk::TransformFileWriter::New();
      
    SDMWriter->SetFileName( fileOutputTransformation );
    SDMWriter->SetInput( SDMTransform   );
    SDMWriter->Update();
  }

  // Write out the deformation field
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  sdmField = SDMTransform->GetSingleDeformationField();

  if ( fileOutputDeformation.length() > 0 ) {

    std::cout << "Get single deformation field ";
    fieldWriter->SetFileName( fileOutputDeformation );
    fieldWriter->SetInput( sdmField );          
    std::cout << "write " << fileOutputDeformation;
    try
      {
	fieldWriter->Update();
      }
    catch( itk::ExceptionObject & excp )
      {
	std::cerr <<"Exception thrown on writing deformation field";
	std::cerr << excp << std::endl; 
      }
  }


  // Prepare error images
  // ~~~~~~~~~~~~~~~~~~~~

  initialerrorImage = InternalImageType::New();
  initialerrorImage->SetRegions(region);
  initialerrorImage->SetOrigin(origin);
  initialerrorImage->SetSpacing(spacing);
  initialerrorImage->SetDirection(direction);
  initialerrorImage->Allocate();
  
  errorImage = InternalImageType::New();
  errorImage->SetRegions(region);
  errorImage->SetOrigin(origin);
  errorImage->SetSpacing(spacing);
  errorImage->SetDirection(direction);
  errorImage->Allocate();
  

  // Iterate through computing error
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InternalPixelType sumDiffSq = 0;

  DisplacementType displacement2;
  DisplacementType difference;

  InternalPixelType error;
  InternalPixelType meanError = 0.0;
  InternalPixelType maxError = 0.0;
  InternalPixelType minError = 1000000.0;
  InternalPixelType stdError = 0.0;
  
  InternalPixelType initialerror;
  InternalPixelType initialmeanError = 0.0;
  InternalPixelType initialmaxError = 0.0;
  InternalPixelType initialminError = 1000000.0;
  InternalPixelType initialstdError = 0.0;
  
  unsigned int N = 0;
  
  PointType pointFieldTransformed;
  PointType pointSDMtransformed;


  for ( itField1.GoToBegin(); !itField1.IsAtEnd(); ++itField1)
    {
      Findex = itField1.GetIndex();
      
      if ((maskName == NULL) || (mask->GetPixel(Findex) == mask_value))
	{
	  
	  targetField->TransformIndexToPhysicalPoint( Findex, pointFieldTransformed );

	  pointSDMtransformed = SDMTransform->TransformPoint( pointFieldTransformed );

	  displacement1 = itField1.Get();                 
	  pointFieldTransformed += displacement1;
          
	  difference = pointSDMtransformed - pointFieldTransformed;

	  std::cout << Findex << " " << pointFieldTransformed << " " << displacement1 << " " << pointSDMtransformed << std::endl;
          
	  initialerror = 0.0;
	  error = 0.0;
	  for (unsigned int m=0; m< ImageDimension; m++ )
	    {
	      initialerror = initialerror + displacement1[m]*displacement1[m];                          
	      error = error + difference[m]*difference[m];                          
	    }
	  initialerror = sqrt(initialerror);
	  error = sqrt(error);
          
	  initialmeanError = initialmeanError + initialerror;
	  meanError = meanError + error;
          
	  if (initialmaxError < initialerror) initialmaxError = initialerror;
	  if (initialminError > initialerror) initialminError = initialerror;
          
	  if (maxError < error) maxError = error;
	  if (minError > error) minError = error;
          
	  N=N+1;
	  std::cout << N << " " << error << " " << minError << " " << meanError/N << " " <<  maxError << std::endl;
          
	  // always store in image
	  initialerrorImage->SetPixel(Findex, initialerror);
	  errorImage->SetPixel(Findex, error);
	}
    }

  initialmeanError = initialmeanError / N;
  meanError = meanError / N;
  
  //
  // second pass, compute standard deviation
  //
  {
    ImageIterator it( initialerrorImage, initialerrorImage->GetLargestPossibleRegion() );

    sumDiffSq = 0;
    if (maskName != NULL)
      {
	MaskIterator itM( mask, mask->GetLargestPossibleRegion() );
	for ( it.GoToBegin(), itM.GoToBegin(); !it.IsAtEnd(); ++it, ++itM)
	  {
	    if (itM.Get() ==  mask_value)
	      {
		initialerror = it.Get();
		sumDiffSq = sumDiffSq + (initialerror - initialmeanError)*(initialerror - initialmeanError);
	      }
	  }
      }
    else
      {
	for ( it.GoToBegin(); !it.IsAtEnd(); ++it)
	  {
	    initialerror = it.Get();
	    sumDiffSq = sumDiffSq + (initialerror - initialmeanError)*(initialerror - initialmeanError);
	  }
      }
    initialstdError = vcl_sqrt(sumDiffSq / (N - 1));
    
    std::cout <<  "Initial min  error: " << niftk::ConvertToString( initialminError  );
    std::cout <<  "Initial mean error: " << niftk::ConvertToString( initialmeanError );
    std::cout <<  "Initial max  error: " << niftk::ConvertToString( initialmaxError  );
    std::cout <<  "Initial std  error: " << niftk::ConvertToString( initialstdError  );
  }


  {
    ImageIterator it( errorImage, errorImage->GetLargestPossibleRegion() );

    sumDiffSq = 0;
    if (maskName != NULL)
      {
	MaskIterator itM( mask, mask->GetLargestPossibleRegion() );
	for ( it.GoToBegin(), itM.GoToBegin(); !it.IsAtEnd(); ++it, ++itM)
	  {
	    if (itM.Get() ==  mask_value)
	      {
		error = it.Get();
		sumDiffSq = sumDiffSq + (error-meanError) * (error-meanError);
	      }
	  }
      }
    else
      {
	for ( it.GoToBegin(); !it.IsAtEnd(); ++it)
	  {
	    error = it.Get();
	    sumDiffSq = sumDiffSq + (error-meanError) * (error-meanError);
	  }
      }
    stdError = vcl_sqrt(sumDiffSq / (N - 1));
    
    std::cout <<  "Registration min  error: " << niftk::ConvertToString( minError  );
    std::cout <<  "Registration mean error: " << niftk::ConvertToString( meanError );
    std::cout <<  "Registration max  error: " << niftk::ConvertToString( maxError  );
    std::cout <<  "Registration std  error: " << niftk::ConvertToString( stdError  );
  }


  // Write the error image to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (outName != NULL)
    {  
      writer->SetFileName( outName );
      writer->SetInput( errorImage );

      std::cout <<  "Writing error image to file: " << outName ;
          
      try 
	{ 
	  writer->Update(); 
	} 
      catch( itk::ExceptionObject & err ) 
	{ 
	  std::cerr << "ExceptionObject caught !" << std::endl; 
	  std::cerr << err << std::endl; 
	  return EXIT_FAILURE;
	}
    }

  return EXIT_SUCCESS;
}




// -------------------------------------------------------------------
// main()
// -------------------------------------------------------------------

int main(int argc, char** argv)
{
  bool flgOptimiseTranslation = false;
  
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_TRANSLATION, flgOptimiseTranslation);

  if (flgOptimiseTranslation) {

    typedef itk::TranslationPCADeformationModelTransform< VectorComponentType, ImageDimension > TransformType;
    return FitPCADeformationModelToDeformationField<TransformType>(argc, argv);

  }
  else {

    typedef itk::PCADeformationModelTransform< VectorComponentType, ImageDimension > TransformType;
    return FitPCADeformationModelToDeformationField<TransformType>(argc, argv);
  }

  return 0;
}


