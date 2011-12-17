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

#include "itkIndex.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "vnl/vnl_math.h"

// for deformation field
#include "itkVector.h"


struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_SWITCH, "dbg", 0, "Output debugging information."},
  {OPT_SWITCH, "v", 0,   "Verbose output during execution."},

  {OPT_STRING|OPT_REQ, "ox", "filename", "Output 'x' component displacement field."},
  {OPT_STRING|OPT_REQ, "oy", "filename", "Output 'y' component displacement field."},
  {OPT_STRING|OPT_REQ, "oz", "filename", "Output 'z' component displacement field."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "Input deformation vector field."},

  {OPT_DONE, NULL, NULL, "Program to generate the 'x', 'y' and 'z' component displacement "
   "fields from a single deformation 3-tuple vector field."}
};


enum {
  O_DEBUG = 0,
  O_VERBOSE,

  O_OUTPUT_X_COMPONENT,
  O_OUTPUT_Y_COMPONENT,
  O_OUTPUT_Z_COMPONENT,

  O_INPUT_DEFORMATION_FIELD
};


// creates deformation field from 3 images containing displacement ux, uy, uz


int main( int argc, char * argv [] )
{
  bool debug;                    // Output debugging information
  bool verbose;                  // Verbose output during execution

  std::string outUX;
  std::string outUY;
  std::string outUZ;

  std::string dofinName;

  // Parse the command line
  // ~~~~~~~~~~~~~~~~~~~~~~
  
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_OUTPUT_X_COMPONENT, outUX);
  CommandLineOptions.GetArgument(O_OUTPUT_Y_COMPONENT, outUY);
  CommandLineOptions.GetArgument(O_OUTPUT_Z_COMPONENT, outUZ);

  CommandLineOptions.GetArgument(O_INPUT_DEFORMATION_FIELD, dofinName);


  /** Typedefs. */
  typedef float PixelType;
  enum {ImageDimension = 3};
  typedef itk::Image<PixelType,ImageDimension> ImageType;  
  typedef itk::Vector<float,ImageDimension> VectorType;
  typedef itk::Image<VectorType,ImageDimension> FieldType;
  typedef itk::Image<VectorType::ValueType,ImageDimension> FloatImageType;
  typedef ImageType::IndexType  IndexType;

  typedef itk::Image< VectorType,  ImageDimension >   DeformationFieldType;

  //--------------------------------------------------------

  // Read deformation image
  typedef itk::ImageFileReader<  DeformationFieldType  > DeformationReaderType;
  
  DeformationReaderType::Pointer reader = DeformationReaderType::New();
  DeformationFieldType::Pointer field = DeformationFieldType::New();
  
  reader->SetFileName( dofinName.c_str() );  
  try
    {
      reader->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
      std::cerr << "Exception thrown by writer" << std::endl;
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }
  field = reader->GetOutput();
  field->DisconnectPipeline();



  IndexType index;
  
  // Creating images
  ImageType::Pointer imageUX = ImageType::New();
  imageUX->SetOrigin( field->GetOrigin() );
  imageUX->SetSpacing( field->GetSpacing() );
  imageUX->SetRegions( field->GetLargestPossibleRegion() );
  imageUX->SetDirection( field->GetDirection() );
  imageUX->Allocate();

  ImageType::Pointer imageUY = ImageType::New();
  imageUY->SetOrigin( field->GetOrigin() );
  imageUY->SetSpacing( field->GetSpacing() );
  imageUY->SetRegions( field->GetLargestPossibleRegion() );
  imageUY->SetDirection( field->GetDirection() );
  imageUY->Allocate();

  ImageType::Pointer imageUZ = ImageType::New();
  imageUZ->SetOrigin( field->GetOrigin() );
  imageUZ->SetSpacing( field->GetSpacing() );
  imageUZ->SetRegions( field->GetLargestPossibleRegion() );
  imageUZ->SetDirection( field->GetDirection() );
  imageUZ->Allocate();



  VectorType displacementVector;


  typedef itk::ImageRegionIteratorWithIndex<ImageType> Iterator;
  Iterator itUX( imageUX, imageUX->GetLargestPossibleRegion() );
  Iterator itUY( imageUY, imageUY->GetLargestPossibleRegion() );
  Iterator itUZ( imageUZ, imageUZ->GetLargestPossibleRegion() );

  std::cout << std::string("Get components of deformation field");
  
  for ( itUX.Begin(), itUY.Begin(), itUZ.Begin(); !itUX.IsAtEnd(); ++itUX,  ++itUY,  ++itUZ )
    {
      index = itUX.GetIndex();      
      displacementVector = field->GetPixel(index);
          
      //if (displacementVector[0] > 1.0) {
      //	printf(" displacementVector %f %f %f\n",displacementVector[0],displacementVector[1],displacementVector[2]);
      //}
          
      itUX.Set(displacementVector[0]);
      itUY.Set(displacementVector[1]);
      itUZ.Set(displacementVector[2]);

    }        
  
  typedef itk::ImageFileWriter< ImageType  > ImageWriterType;
  ImageWriterType::Pointer  writer  = ImageWriterType::New();

  writer->SetInput(imageUX);
  writer->SetFileName(  outUX.c_str() );
  try
    {
      writer->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
      std::cerr <<"Exception thrown by 'x' writer";
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }
  imageUX->DisconnectPipeline();
 

  writer->SetInput(imageUY);
  writer->SetFileName(  outUY.c_str() );
  try
    {
      writer->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
      std::cerr <<"Exception thrown by 'y' writer";
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }
  imageUY->DisconnectPipeline();

  writer->SetInput(imageUZ);
  writer->SetFileName( outUZ.c_str() );
  try
    {
      writer->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
      std::cerr <<"Exception thrown by 'z' writer";
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }
  imageUZ->DisconnectPipeline();

  return EXIT_SUCCESS;
}
