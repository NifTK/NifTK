/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: jhh                 $
 $Date:: 2011-12-16 13:12:13 +#$
 $Rev:: 8041                   $

 Copyright (c) UCL : See the file NifTKCopyright.txt in the top level 
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "LogHelper.h"
#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "vnl/vnl_matrix_fixed.h"
#include "vnl/vnl_math.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkVector.h"
#include "itkLightProcessObject.h"
#include "itkTextOutput.h"

#include "itkImagePCAShapeModelEstimator.h"
#include "itkAddImageFilter.h"
#include "itkMultiplyByConstantImageFilter.h"

#include "itkVectorImageToImageAdaptor.h"
#include "itkImageToVectorImageFilter.h"

#include "itkIndex.h"
#include "itkImageRegionIteratorWithIndex.h"

//Data definitions 
#define   NDIMENSION          3

// class to support progress feeback

struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_SWITCH, "dbg", 0, "Output debugging information."},
  {OPT_SWITCH, "v", 0,   "Verbose output during execution."},

  {OPT_STRING|OPT_REQ, "o", "filename", "Output PCA deformation components."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "Individual multiple deformation fields."},
  {OPT_MORE, NULL, "...", NULL},

  {OPT_DONE, NULL, NULL, "Program to create a PCA model of eigen-deformations."}
};


enum {
  O_DEBUG = 0,
  O_VERBOSE,

  O_OUTPUT_PCA_FILE,

  O_DEFORMATION_FIELD,
  O_MORE
};


int main(int argc, char * argv[] )
{
  bool debug;                    // Output debugging information
  bool verbose;                  // Verbose output during execution

  char *fileDeformationField = 0; // A mandatory character string argument
  char **fileDeformationFields = 0;	  // Multiple character string arguments

  int i;			// Loop counter
  int arg;			// Index of arguments in command line 

  int numberOfInputDeformations = 0; // The number of input deformations

  std::string fileOutputPCA;	// The output PCA file

  // This reads logging configuration from log4cplus.properties

  log4cplus::LogLevel logLevel = log4cplus::WARN_LOG_LEVEL;

  niftk::LogHelper::SetupBasicLogging();


  // Parse the command line
  // ~~~~~~~~~~~~~~~~~~~~~~
  
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  if (CommandLineOptions.GetArgument(O_VERBOSE, verbose)) {
    if (logLevel > log4cplus::INFO_LOG_LEVEL)
      logLevel = log4cplus::INFO_LOG_LEVEL;
  }

  if (CommandLineOptions.GetArgument(O_DEBUG, debug)) {
    if (logLevel > log4cplus::DEBUG_LOG_LEVEL)
      logLevel = log4cplus::DEBUG_LOG_LEVEL;
  }
  niftk::LogHelper::SetLogLevel(logLevel);

  CommandLineOptions.GetArgument(O_OUTPUT_PCA_FILE, fileOutputPCA);

  CommandLineOptions.GetArgument(O_DEFORMATION_FIELD, fileDeformationField);

  // Call the 'OPT_MORE' option to determine the position of the list
  // of extra command line options ('arg').
  CommandLineOptions.GetArgument(O_MORE, arg);
  
  if (arg < argc) {		   // Many deformation fields
    numberOfInputDeformations = argc - arg + 1;
    fileDeformationFields = &argv[arg-1];

    niftk::LogHelper::InfoMessage(std::string("Deformation fields: "));
    for (i=0; i<numberOfInputDeformations; i++)
      niftk::LogHelper::InfoMessage( niftk::ConvertToString( (int) i+1) + " " + fileDeformationFields[i] );
  }
  else if (fileDeformationField) { // Single deformation field
    numberOfInputDeformations = 1;
    fileDeformationFields = &fileDeformationField;

    niftk::LogHelper::InfoMessage(std::string("Deformation field: ") + fileDeformationFields[0] );
  }
  else {
    numberOfInputDeformations = 0;
    fileDeformationFields = 0;
  }

  itk::OutputWindow::SetInstance(itk::TextOutput::New().GetPointer());

  typedef double ImagePixelType;
  typedef itk::Vector< ImagePixelType, NDIMENSION > VectorPixelType; 
  typedef itk::Image< VectorPixelType, NDIMENSION > InputDeformationFieldType;
  typedef itk::Image< VectorPixelType, NDIMENSION > OutputDeformationFieldType;

  typedef itk::Image< ImagePixelType, NDIMENSION > InputImageType; 
  typedef itk::Image< ImagePixelType, NDIMENSION > OutputImageType;

  typedef OutputImageType::IndexType  IndexType;
  IndexType index;
  typedef OutputDeformationFieldType::IndexType  FieldIndexType;
  FieldIndexType Findex;


  typedef itk::VectorImageToImageAdaptor< InputDeformationFieldType, NDIMENSION >   ImageAdaptorType;
  typedef itk::ImageToVectorImageFilter<OutputImageType> ImageToVectorImageFilterType;

  typedef itk::MultiplyByConstantImageFilter< OutputImageType, ImagePixelType, OutputImageType  >   FilterType;

  ImageToVectorImageFilterType::Pointer imageToVectorImageFilter = ImageToVectorImageFilterType::New();
  
  InputDeformationFieldType::RegionType region;
  InputDeformationFieldType::SizeType size;
  InputDeformationFieldType::SpacingType spacing;
  InputDeformationFieldType::DirectionType  direction;
  InputDeformationFieldType::PointType origin;

  //----------------------------------------------------------------------
  // Read deformation fields
  //----------------------------------------------------------------------
  const unsigned int NUMTRAINIMAGES = numberOfInputDeformations;
  const unsigned int NUMLARGESTPC = NUMTRAINIMAGES - 1;  

  std::vector<InputDeformationFieldType::Pointer> fields( NUMTRAINIMAGES );
  std::vector<OutputDeformationFieldType::Pointer> outFields( NUMLARGESTPC + 1);

  // CHANGED: outimages are as many as 'NUMLARGESTPC + 1' and as big as (x,y,3*z)
  std::vector<InputImageType::Pointer> images( NUMTRAINIMAGES );
  std::vector<OutputImageType::Pointer> outImages( NUMLARGESTPC + 1 );//(NUMLARGESTPC + 1)*NDIMENSION );

  for (unsigned int k=0; k< NUMTRAINIMAGES; k++ )
    {
      typedef   itk::ImageFileReader< InputDeformationFieldType >  FieldReaderType;
      FieldReaderType::Pointer reader = FieldReaderType::New();
      reader->SetFileName( fileDeformationFields[k] );
      try
	{
	  reader->Update();
	}
      catch( itk::ExceptionObject & exp )
	{
	  niftk::LogHelper::ErrorMessage("Exception thrown while reading the input file.");
	  std::cerr << exp << std::endl; 
	  return EXIT_FAILURE;
	}
      fields[k] = reader->GetOutput();
          
      region =  fields[k]->GetLargestPossibleRegion();
      spacing = fields[k]->GetSpacing();
      size =    region.GetSize();
      direction = fields[k]->GetDirection();
      origin = fields[k]->GetOrigin();
          
      niftk::LogHelper::InfoMessage( niftk::ConvertToString((int) k) 
				     + " Deformation field spacing: " 
				     + niftk::ConvertToString(spacing[0]) + ", " 
				     + niftk::ConvertToString(spacing[1]) + ", " 
				     + niftk::ConvertToString(spacing[2]));
      niftk::LogHelper::InfoMessage( std::string(" size: ")
				     + niftk::ConvertToString(size[0]) + ", " 
				     + niftk::ConvertToString(size[1]) + ", " 
				     + niftk::ConvertToString(size[2]));
      
      images[k] = InputImageType::New();
      InputImageType::RegionType concatInputRegion;
      InputImageType::SizeType concatInputSize = size;
      concatInputSize[2] = 3*concatInputSize[2];
      concatInputRegion.SetSize( concatInputSize );
      InputImageType::RegionType::IndexType regionInputStart;
      regionInputStart[0] = 0;
      regionInputStart[1] = 0;
      regionInputStart[2] = 0;
      concatInputRegion.SetIndex( regionInputStart );
      images[k]->SetRegions(concatInputRegion);
      images[k]->SetSpacing(spacing);
      images[k]->SetOrigin(origin);
      images[k]->Allocate();
    }
  
  //----------------------------------------------------------------------
  // Set the image model estimator
  //----------------------------------------------------------------------

  typedef itk::ImagePCAShapeModelEstimator<InputImageType, OutputImageType> 
    ImagePCAShapeModelEstimatorType;

  ImagePCAShapeModelEstimatorType::Pointer 
    applyPCAShapeEstimator = ImagePCAShapeModelEstimatorType::New();

  //----------------------------------------------------------------------
  // Set the parameters of the clusterer
  //----------------------------------------------------------------------
  applyPCAShapeEstimator->SetNumberOfTrainingImages( NUMTRAINIMAGES );
  applyPCAShapeEstimator->SetNumberOfPrincipalComponentsRequired( NUMLARGESTPC );


  for(unsigned int i= 0; i<= NUMLARGESTPC; i++ )
    {
      outFields[i] = OutputDeformationFieldType::New();
      outFields[i]->SetRegions(region);
      outFields[i]->SetSpacing(spacing);
      outFields[i]->SetOrigin(origin);
      outFields[i]->Allocate();
 
      // TO CHANGE: outImages will now be as many as the NUMLARGESTPC and bigger in size
      OutputImageType::RegionType concatRegion;
      OutputImageType::SizeType concatSize = size;//inputImage->GetLargestPossibleRegion().GetSize();
      concatSize[2] = 3*concatSize[2];
      concatRegion.SetSize( concatSize );
      OutputImageType::RegionType::IndexType regionStart;
      regionStart[0] = 0;
      regionStart[1] = 0;
      regionStart[2] = 0;
      concatRegion.SetIndex( regionStart );
 
      outImages[i] = OutputImageType::New();
      outImages[i]->SetRegions(concatRegion);
      outImages[i]->SetSpacing(spacing);
      outImages[i]->SetOrigin(origin);
      outImages[i]->Allocate();

      // Display for debugging
      //std::cout << "The original size is: " << size[0] << " " << size[1] << " " << size[2] << std::endl;
      //std::cout << "and the concatenated: " << concatSize[0] << " " << concatSize[1] << " " << concatSize[2] << std::endl;

    }
  
  VectorPixelType displacementVector;
  
  typedef itk::ImageRegionIteratorWithIndex<OutputDeformationFieldType> FieldIterator;
  typedef itk::ImageRegionIteratorWithIndex<OutputImageType> Iterator;

  InputImageType::SizeType inputSize = size;
 
  std::cout << "Before the loop to assign the training images to the estimator..." << std::endl;

  // CHANGED: do 1 PCA and not 3 for dx, dy and dz
  // do PCA for each individual component of vector field          
  for (unsigned int k=0; k< NUMTRAINIMAGES; k++ )
    {
      images[k]->DisconnectPipeline();

      FieldIterator itField( fields[k], fields[k]->GetLargestPossibleRegion() ); 
      //Iterator itImage( images[k], images[k]->GetLargestPossibleRegion() );
                  
      for ( itField.Begin(); !itField.IsAtEnd(); ++itField) 
        {
	  Findex = itField.GetIndex();	    
          displacementVector = fields[k]->GetPixel( Findex );//itField.Get();//
	  /*if ((Findex[0]==124)&&(Findex[1]==134)&&(Findex[2]==50))
	  {
	    std::cout << "at itField=10, the deformation field is: "<<displacementVector[0]<<" "
	              <<displacementVector[1]<<" "<<displacementVector[2]<<std::endl;
	  }*/
	  for (unsigned int m=0; m< NDIMENSION; m++ )
	    {
	      images[k]->SetPixel( Findex, displacementVector[m] );                          
	      Findex[2] += inputSize[2];
	    }  
        }
      //images[k]->Update();
      applyPCAShapeEstimator->SetInput(k, images[k]);
    }
  applyPCAShapeEstimator->Update();

  std::cout << "After the estimator is updated..." << std::endl;
          
  typedef ImagePCAShapeModelEstimatorType::Superclass GenericEstimatorType;

  std::cout << "Getting the eigenvalues..." << std::endl;  

  //Print the eigen vectors
  vnl_vector<ImagePixelType> eigenValues = applyPCAShapeEstimator->GetEigenValues();
          
  niftk::LogHelper::InfoMessage( std::string("Creating PCA... ") );
          

  //Print out the number of training images and the number of principal components
  niftk::LogHelper::InfoMessage( std::string("The number of training images are: ")
               			 + niftk::ConvertToString( applyPCAShapeEstimator->GetNumberOfTrainingImages() ));
  niftk::LogHelper::InfoMessage( std::string("The ")
				 + niftk::ConvertToString( applyPCAShapeEstimator->GetNumberOfPrincipalComponentsRequired() ) 
				 + " largest eigen values are:" );
 
  double sumEigenValues = 0.0;                  
  double cumEigenValues = 0.0;

  for(unsigned int i= 0; i< NUMLARGESTPC; i++ )
    {
      sumEigenValues += sqrt( eigenValues[i] ); 
      std::cout << "sqrt of eigenvalue " << i << " is: " << sqrt( eigenValues[i] ) << std::endl;
    }
                                    
  for(unsigned int i= 0; i< NUMLARGESTPC; i++ )
    {
      cumEigenValues += sqrt( eigenValues[i] );
                          
      niftk::LogHelper::InfoMessage( niftk::ConvertToString(i+1) + " " + niftk::ConvertToString(sqrt(eigenValues[ i ])) + " " 
         			     + niftk::ConvertToString(sqrt(eigenValues[ i ])/sumEigenValues*100) + " % " 
				     + niftk::ConvertToString(cumEigenValues/sumEigenValues*100) + " %"); 
    }  
 
  std::cout << "Getting all the eigenvector images.. " << std::endl;
          
  // get all results              
  for (unsigned int i= 0; i<= NUMLARGESTPC; i++ )
    {
      outImages[i]->DisconnectPipeline();
      if (i == 0) 
        {
          // MeanImage
          outImages[i] = applyPCAShapeEstimator->GetOutput( i );
        }
      else
        {
	  FilterType::Pointer multiplyfilter = FilterType::New();            
	  multiplyfilter->SetInput( applyPCAShapeEstimator->GetOutput( i ) );
	  multiplyfilter->SetConstant( sqrt(eigenValues[ i-1 ]) );
	  outImages[i] = multiplyfilter->GetOutput();
	  multiplyfilter->Update();
        }
      outImages[i]->DisconnectPipeline();
      }
  
  for (unsigned int i= 0; i<= NUMLARGESTPC; i++ )
    { 
      FieldIterator itField( outFields[i], outFields[i]->GetLargestPossibleRegion() );
      //Iterator itImage( outImages[i], outImages[i]->GetLargestPossibleRegion() );
          
      for ( itField.Begin(); !itField.IsAtEnd(); ++itField )
        {
          Findex = itField.GetIndex();
   	  for (unsigned int m=0; m< NDIMENSION; m++ )
	    {
	      displacementVector[m] = outImages[i]->GetPixel( Findex );                          
	      Findex[2] += inputSize[2];
	    }       
	  itField.Set(displacementVector);
        }
          
      std::cout << "Writing the eigenvector images.. " << std::endl;
      
      // output
      typedef itk::ImageFileWriter < OutputDeformationFieldType >  WriterType;
      WriterType::Pointer writer = WriterType::New();

      std::stringstream sstr;
      sstr << fileOutputPCA << "_C" << i << ".mha";
                  
      writer->SetFileName( sstr.str() );
      writer->SetInput( outFields[i] );

      try{
          writer->Update();
        }
      catch( itk::ExceptionObject & excp ) {
	  niftk::LogHelper::ErrorMessage("Exception caught");
	  std::cerr << excp << std::endl;
	  return EXIT_FAILURE;
        }         

      sstr.str("");
    }
  
  return EXIT_SUCCESS;
}
