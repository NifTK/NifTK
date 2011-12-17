/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: ad                  $
 $Date:: 2011-09-20 14:34:44 +#$
 $Rev:: 7333                   $

 Copyright (c) UCL : See the file LICENSE.txt in the top level
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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

  // Parse the command line
  // ~~~~~~~~~~~~~~~~~~~~~~
  
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_OUTPUT_PCA_FILE, fileOutputPCA);

  CommandLineOptions.GetArgument(O_DEFORMATION_FIELD, fileDeformationField);

  // Call the 'OPT_MORE' option to determine the position of the list
  // of extra command line options ('arg').
  CommandLineOptions.GetArgument(O_MORE, arg);
  
  if (arg < argc) {		   // Many deformation fields
    numberOfInputDeformations = argc - arg + 1;
    fileDeformationFields = &argv[arg-1];

    std::cout << "Deformation fields: ";
    for (i=0; i<numberOfInputDeformations; i++)
      std::cout <<  niftk::ConvertToString( (int) i+1) << " " << fileDeformationFields[i];
  }
  else if (fileDeformationField) { // Single deformation field
    numberOfInputDeformations = 1;
    fileDeformationFields = &fileDeformationField;

    std::cout << "Deformation field: " << fileDeformationFields[0];
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

  std::vector<InputImageType::Pointer> images( NUMTRAINIMAGES );
  std::vector<OutputImageType::Pointer> outImages( NUMLARGESTPC + 1 );

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
	  std::cerr << "Exception thrown while reading the input file.";
	  std::cerr << exp << std::endl; 
	  return EXIT_FAILURE;
	}
      fields[k] = reader->GetOutput();
          
      region =  fields[k]->GetLargestPossibleRegion();
      spacing = fields[k]->GetSpacing();
      size =    region.GetSize();
      direction = fields[k]->GetDirection();
      origin = fields[k]->GetOrigin();
          
      std::cout <<  niftk::ConvertToString((int) k)
				     << " Deformation field spacing: "
				     << niftk::ConvertToString(spacing[0]) << ", "
				     << niftk::ConvertToString(spacing[1]) << ", "
				     << niftk::ConvertToString(spacing[2]);
      std::cout <<  " size: "
         << niftk::ConvertToString(size[0]) << ", "
         << niftk::ConvertToString(size[1]) << ", "
         << niftk::ConvertToString(size[2]);
      
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
 
      OutputImageType::RegionType concatRegion;
      OutputImageType::SizeType concatSize = size;
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
  }
  
  VectorPixelType displacementVector;
  
  typedef itk::ImageRegionIteratorWithIndex<OutputDeformationFieldType> FieldIterator;
  typedef itk::ImageRegionIteratorWithIndex<OutputImageType> Iterator;

  InputImageType::SizeType inputSize = size;
 
  // reshape the input images and do 1 PCA       
  for (unsigned int k=0; k< NUMTRAINIMAGES; k++ )
    {
      images[k]->DisconnectPipeline();

      FieldIterator itField( fields[k], fields[k]->GetLargestPossibleRegion() ); 
                 
      for ( itField.Begin(); !itField.IsAtEnd(); ++itField) 
        {
	  Findex = itField.GetIndex();	    
          displacementVector = fields[k]->GetPixel( Findex );
	  for (unsigned int m=0; m< NDIMENSION; m++ )
	    {
	      images[k]->SetPixel( Findex, displacementVector[m] );                          
	      Findex[2] += inputSize[2];
	    }  
        }
      applyPCAShapeEstimator->SetInput(k, images[k]);
    }
  applyPCAShapeEstimator->Update();
        
  typedef ImagePCAShapeModelEstimatorType::Superclass GenericEstimatorType;

  //Print the eigen vectors
  vnl_vector<ImagePixelType> eigenValues = applyPCAShapeEstimator->GetEigenValues();
          
  std::cout << "Creating PCA... ";
          

  //Print out the number of training images and the number of principal components
  std::cout <<  "The number of training images are: "
               			 + niftk::ConvertToString( applyPCAShapeEstimator->GetNumberOfTrainingImages() );
  std::cout <<  "The "
				 << niftk::ConvertToString( applyPCAShapeEstimator->GetNumberOfPrincipalComponentsRequired() )
				 << " largest eigen values are:";
 
  double sumEigenValues = 0.0;                  
  double cumEigenValues = 0.0;

  for(unsigned int i= 0; i< NUMLARGESTPC; i++ )
    {
      sumEigenValues += sqrt( eigenValues[i] ); 
    }
                                    
  for(unsigned int i= 0; i< NUMLARGESTPC; i++ )
    {
      cumEigenValues += sqrt( eigenValues[i] );
                          
      std::cout <<  niftk::ConvertToString(i+1) << " " << niftk::ConvertToString(sqrt(eigenValues[ i ])) << " "
         			     << niftk::ConvertToString(sqrt(eigenValues[ i ])/sumEigenValues*100) << " % "
				     << niftk::ConvertToString(cumEigenValues/sumEigenValues*100) << " %";
    }  
        
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
	  // multiply by squareroot of eigenvalue to normalize
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
	  std::cerr << "Exception caught";
	  std::cerr << excp << std::endl;
	  return EXIT_FAILURE;
        }         

      sstr.str("");
    }
  
  return EXIT_SUCCESS;
}
