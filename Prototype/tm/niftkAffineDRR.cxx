/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date:  $
 Revision          : $Revision:  $
 Last modified by  : $Author:  $

 Original author   : t.mertzanidou@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <iomanip>

#include "itkAffineTransform2D3D.h"
#include "itkEulerAffineTransform.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkPerspectiveProjectionTransform.h"
#include "itkRayCastInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkTransformFactory.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"
#include "itkNIFTKTransformIOFactory.h"

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

//#define WRITE_CUBE_IMAGE_TO_FILE


// -------------------------------------------------------------------------
// Command line parameters
// -------------------------------------------------------------------------

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output [no]."},

  {OPT_FLOATx2, "res", "dx,dy", "Pixel spacing of the output image in mm [1x1]."},
  {OPT_INTx2, "size", "nx,ny", "Dimension of the output image in pixels [501x501]"},

  {OPT_FLOAT, "sid", "distance", "The focal length or source-to-image distance of the X-ray set in mm [660]."},

  {OPT_STRING, "params", "filename", "An input file containing all the transform parameters."},
  {OPT_STRING, "transform", "filename", "An input file containing the ITK affine transformation."},
  {OPT_STRING, "persp", "filename", "An input file containing the ITK perspective transformation."},

  {OPT_FLOATx3, "t", "tx,ty,tz", "Translation of the 3D volume from the detector at (0,0,SID) towards the source in mm [(0,0,160.)]."},

  {OPT_FLOAT, "rx", "degrees", "Rotation around 'x' axis in degrees [0]"},
  {OPT_FLOAT, "ry", "degrees", "Rotation around 'y' axis in degrees [0]"},
  {OPT_FLOAT, "rz", "degrees", "Rotation around 'z' axis in degrees [0]"},

  {OPT_FLOATx2, "normal", "px,py", "The 2D projection normal position in mm [0x0]"},

  {OPT_FLOATx3, "cor", "cx,cy,cz", "The centre of rotation relative to centre of volume in mm [(0,0,0)]"},

  {OPT_DOUBLE, "threshold", "value", "Intensity projection threshold [0]."},

  {OPT_STRING, "ota", "filename", "Output the affine transformation."},
  {OPT_STRING, "ote", "filename", 
   "Output the forward affine transformation wrt. the origin at the source. "
   "This is used to transform points in the 3D volume to 2D."},

  {OPT_STRING, "otp", "filename", "Output the perspective transformation."},

  {OPT_STRING, "o", "filename", "The output registered (transformed and projected) image file."},

  {OPT_STRING|OPT_REQ|OPT_LONELY, NULL, "filename", "The input volume."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to create a Digitally Reconstructed Radiograph from a 3D volume.\n"
  }
};


enum {
  O_VERBOSE,

  O_RESOLUTION_2D,
  O_SIZE_2D,

  O_SID,

  O_PARAMETERS_FILE,
  O_TRANSFORMATION_FILE,
  O_PERSPECTIVE_FILE,

  O_TRANSLATE_VOLUME,

  O_ROTATION_IN_X,
  O_ROTATION_IN_Y,
  O_ROTATION_IN_Z,

  O_NORMAL_POSITION,

  O_CENTRE_OF_ROTATION,

  O_THRESHOLD,

  O_OUTPUT_AFFINE_TRANSFORMATION,
  O_OUTPUT_TRANSFORMATION_WITH_ORIGIN_AT_SOURCE,

  O_OUTPUT_PERSPECTIVE_TRANSFORMATION,

  O_OUTPUT_DRR,
  O_INPUT_VOLUME
};


int main( int argc, char ** argv )
{
  char *fileInputVolume = NULL;

  char *fileInputParams = NULL;
  char *fileInputTransformation = NULL;
  char *fileInputPerspective = NULL;

  char *fileOutputAffineTransWithOriginAtSource = NULL;
  char *fileOutputAffineTransformation = NULL;
  char *fileOutputPerspectiveTransformation = NULL;

  char *fileOutputDRR = NULL;

  bool verbose = false;

  float rx = 0.;
  float ry = 0.;
  float rz = 0.;

  float cx = 0.;
  float cy = 0.;
  float cz = 0.;

  float sid = 660;

  float sx = 1.;
  float sy = 1.;

  int dx = 501;
  int dy = 501;

  int *size2D = 0;

  float o2Dx = 0;
  float o2Dy = 0;

  float *normal2D = 0;
  float *resolution2D = 0;

  float *centreOfRotation = 0;
  float *translateVolume = 0;

  double threshold=0;

   // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);
 

  CommandLineOptions.GetArgument(O_VERBOSE, verbose);

  if (CommandLineOptions.GetArgument(O_RESOLUTION_2D, resolution2D)) {
    sx = resolution2D[0];
    sy = resolution2D[1];
  }

  if (CommandLineOptions.GetArgument(O_SIZE_2D, size2D)) {
    dx = size2D[0];
    dy = size2D[1];
  }

  if ( CommandLineOptions.GetArgument(O_SID, sid) && fileInputPerspective ) {

    std::cerr << "ERROR: Command line options '-sid' and '-persp' are mutually exclusive" << std::endl;
    return EXIT_FAILURE;
  }


  CommandLineOptions.GetArgument(O_PARAMETERS_FILE, fileInputParams);
  CommandLineOptions.GetArgument(O_TRANSFORMATION_FILE, fileInputTransformation);
  CommandLineOptions.GetArgument(O_PERSPECTIVE_FILE, fileInputPerspective);

  if ( fileInputParams && fileInputTransformation ) {

    std::cerr << "ERROR: Command line options '-params' and '-transform' are mutually exclusive" << std::endl;
    return EXIT_FAILURE;
  }
 


  if (! CommandLineOptions.GetArgument(O_TRANSLATE_VOLUME, translateVolume)) {
    translateVolume = new float[3];

    // Set the default
    translateVolume[0] = 0.;
    translateVolume[1] = 0.;
    translateVolume[2] = 160.;
  }
  else if ( fileInputParams || fileInputTransformation ) {

    std::cerr << "ERROR: Command line options '-t' and either '-params' or '-transform' are mutually exclusive" << std::endl;
    return EXIT_FAILURE;
  }

  if ( CommandLineOptions.GetArgument(O_ROTATION_IN_X, rx) &&
       (fileInputParams || fileInputTransformation) ) {

    std::cerr << "ERROR: Command line options '-rx' and either '-params' or '-transform' are mutually exclusive" << std::endl;
    return EXIT_FAILURE;
  }

  if ( CommandLineOptions.GetArgument(O_ROTATION_IN_Y, ry) &&
       (fileInputParams || fileInputTransformation) ) {

    std::cerr << "ERROR: Command line options '-ry' and either '-params' or '-transform' are mutually exclusive" << std::endl;
    return EXIT_FAILURE;
  }

  if ( CommandLineOptions.GetArgument(O_ROTATION_IN_Z, rz) &&
       (fileInputParams || fileInputTransformation) ) {

    std::cerr << "ERROR: Command line options '-rz' and either '-params' or '-transform' are mutually exclusive" << std::endl;
    return EXIT_FAILURE;
  }


  if (CommandLineOptions.GetArgument(O_NORMAL_POSITION, normal2D)) {
    if ( fileInputPerspective ) {

      std::cerr << "ERROR: Command line options '-normal' and '-persp' are mutually exclusive" << std::endl;
      return EXIT_FAILURE;
    }
   
    o2Dx = normal2D[0];
    o2Dy = normal2D[1];
  }

  if (CommandLineOptions.GetArgument(O_CENTRE_OF_ROTATION, centreOfRotation)) {
    if ( fileInputParams || fileInputTransformation ) {

      std::cerr << "ERROR: Command line options '-cor' and either '-params' or '-transform' are mutually exclusive" << std::endl;
      return EXIT_FAILURE;
    }

    cx = centreOfRotation[0];
    cy = centreOfRotation[1];
    cz = centreOfRotation[2];
  }

  CommandLineOptions.GetArgument(O_THRESHOLD, threshold);

  CommandLineOptions.GetArgument(O_OUTPUT_TRANSFORMATION_WITH_ORIGIN_AT_SOURCE, fileOutputAffineTransWithOriginAtSource

); 
  CommandLineOptions.GetArgument(O_OUTPUT_AFFINE_TRANSFORMATION, fileOutputAffineTransformation);
    

  CommandLineOptions.GetArgument(O_OUTPUT_PERSPECTIVE_TRANSFORMATION, fileOutputPerspectiveTransformation);

  CommandLineOptions.GetArgument(O_OUTPUT_DRR, fileOutputDRR);
  CommandLineOptions.GetArgument(O_INPUT_VOLUME, fileInputVolume);


  if (verbose) 
    {
      if (fileInputVolume)  std::cout << "Input image: "  << fileInputVolume  << endl;
      if (fileOutputDRR) std::cout << "Output image: " << fileOutputDRR << endl;
    }

  itk::ObjectFactoryBase::RegisterFactory(itk::NIFTKTransformIOFactory::New());



  // Although we generate a 2D projection of the 3D volume for the
  // purposes of the interpolator both images must be three dimensional.

  const     unsigned int   Dimension = 3;
  typedef   float  InputPixelType; //double
  typedef   float  OutputPixelType; // short
  
  typedef itk::Image< InputPixelType,  Dimension >   InputImageType;
  typedef itk::Image< OutputPixelType, Dimension >   OutputImageType;

  InputImageType::Pointer image;

  // For the purposes of this example we assume the input volume has
  // been loaded into an itk::Image image.

  if (fileInputVolume) 
    {

      typedef itk::ImageFileReader< InputImageType >  ReaderType;
      ReaderType::Pointer reader = ReaderType::New();
      reader->SetFileName( fileInputVolume );

      try 
	{ 
	  reader->Update();
	} 
    
      catch( itk::ExceptionObject & err ) 
	{ 
	  std::cerr << "ERROR: ExceptionObject caught !" << endl; 
	  std::cerr << err << endl; 
	  return -1;
	} 

      image = reader->GetOutput();
    }


  // Print out the details of the input volume

  const itk::Vector<double, 3> imSpacing3D = image->GetSpacing();  
  InputImageType::RegionType   imRegion3D  = image->GetBufferedRegion();
  const itk::Point<double, 3>  imOrigin3D  = image->GetOrigin();

  if (verbose) 
    {
      unsigned int i;
      std::cout << endl << "Input ";

      imRegion3D.Print(std::cout);
    
      std::cout << "  Resolution: [";
      for (i=0; i<Dimension; i++) 
	{
	  std::cout << imSpacing3D[i];
	  if (i < Dimension-1) std::cout << ", ";
	}
      std::cout << "]" << endl;
    
      std::cout << "  Origin: [";
      for (i=0; i<Dimension; i++) 
	{
	  std::cout << imOrigin3D[i];
	  if (i < Dimension-1) std::cout << ", ";
	}
      std::cout << "]" << endl<< endl;
    }

  ////////////////////////////////////////////////////////////////////
  //
  // Creation of a 'ResampleImageFilter' enables coordinates for
  // each of the pixels in the DRR image to be generated. These
  // coordinates are used by the 'RayCastInterpolateImageFunction'
  // to determine the equation of each corresponding ray which is cast
  // through the input volume.
  //
  ////////////////////////////////////////////////////////////////////

  typedef itk::ResampleImageFilter<InputImageType, OutputImageType > FilterType;

  FilterType::Pointer filter = FilterType::New();

  filter->SetInput( image );
  filter->SetDefaultPixelValue( 0 );

  // An Euler transformation is defined to position the input volume.
  // The 'ResampleImageFilter' uses this transform to position the
  // output DRR image for the desired view.

  typedef itk::AffineTransform2D3D< double, 3 >  AffineTransformType;

  AffineTransformType::Pointer affineTransform2D3D = AffineTransformType::New();

  //////////
  double origin3D[ Dimension ];

  const itk::Vector<double, 3> resolution3D = image->GetSpacing();

  typedef InputImageType::RegionType     InputImageRegionType;
  typedef InputImageRegionType::SizeType InputImageSizeType;

  InputImageRegionType imRegion = image->GetBufferedRegion();
  InputImageSizeType   size3D   = imRegion.GetSize();

  origin3D[0] = imOrigin3D[0] + resolution3D[0]*((double) size3D[0] - 1.)/2.; 
  origin3D[1] = imOrigin3D[1] + resolution3D[1]*((double) size3D[1] - 1.)/2.; 
  origin3D[2] = imOrigin3D[2] + resolution3D[2]*((double) size3D[2] - 1.)/2.;

  if (verbose) 
    std::cout << "3D volume:" << std::endl
              << "   dimensions: "
	      << size3D[0] << ", " 
              << size3D[1] << ", " 
              << size3D[2] << " voxels, "
              << std::endl
              << "   size: " 
	      << size3D[0]*resolution3D[0] << ", " 
              << size3D[1]*resolution3D[1] << ", " 
              << size3D[2]*resolution3D[2] << " mm" 
              << std::endl
	      << "   resolution: "
	      << resolution3D[0] << ", " 
              << resolution3D[1] << ", " 
              << resolution3D[2] << std::endl
	      << "   origin: "
	      << origin3D[0] << ", " 
              << origin3D[1] << ", " 
              << origin3D[2] << std::endl << std::endl;

  // Read the affine transformation from an ITK transform file

  if ( fileInputTransformation ) {

    itk::TransformFactory< AffineTransformType >::RegisterTransform();

    typedef itk::TransformFileReader TransformFileReaderType;
    TransformFileReaderType::Pointer reader = TransformFileReaderType::New();

    reader->SetFileName( fileInputTransformation );

    try
      {
	reader->Update();
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << "Exception reading itk::AffineTransform2D3D< double, 3 >" << std::endl;
	std::cerr << err << std::endl;
	return EXIT_FAILURE;
      }

    typedef itk::TransformFileReader::TransformListType * TransformListType;
    TransformListType transforms = reader->GetTransformList();
    
    std::cout << "Number of transforms = " << transforms->size() << std::endl;
    
    itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();
    
    if( ! strcmp((*it)->GetNameOfClass(),"AffineTransform2D3D")) {
      
      affineTransform2D3D = static_cast< AffineTransformType* >((*it).GetPointer());
    }
    else {
      
      std::cerr << "Failed to read transformation from: " << fileInputTransformation;
      return EXIT_FAILURE; 
    }
  }

  // Read the transform parameters from a file and pass on to the transform
  
  else if ( fileInputParams ) {

    typedef AffineTransformType::ParametersType ParametersType;
    ParametersType parameters (12); 

    int i = 0; // counter to know the position in the file
    float x; // variable for reading the file params
  
    ifstream inFile;
    inFile.open( fileInputParams );
    if (!inFile) 
      {
	std::cout << "Unable to open file";
	exit(1); // terminate with error
      }
    while (inFile >> x) 
      {
	if ( i < 12 )
	  {
	    parameters[i] = x;
	    
	    if (verbose) 
	      std::cout << "Parameter: " << i << " = " << parameters[i] << std::endl;

	    i++;
	  }
	else
	  break;
      }

    affineTransform2D3D->SetCenter( origin3D );

    affineTransform2D3D->SetParameters( parameters );
  }
  else {
    
    AffineTransformType::OutputVectorType translateOrigin;
    
    translateOrigin[0] = -origin3D[0];// tx;
    translateOrigin[1] = -origin3D[1];// ty;
    translateOrigin[2] = -origin3D[2];// tz;
    
    affineTransform2D3D->Translate( translateOrigin );  

    AffineTransformType::OutputVectorType angle;

    angle[0] = M_PI/180.0*rx;
    angle[1] = M_PI/180.0*ry;
    angle[2] = M_PI/180.0*rz;
    
    affineTransform2D3D->Rotate( angle );
 
    AffineTransformType::InputPointType center;

    center[0] = cx + origin3D[0];
    center[1] = cy + origin3D[1];
    center[2] = cz + origin3D[2];
    
    affineTransform2D3D->SetCenter(center);
  
    if (verbose) 
      std::cout << "Center: " 
                << center[0] << ", " 
                << center[1] << ", " 
                << center[2] << std::endl << std::endl;
  }

  if (verbose) 
    std::cout << "Transform: " << affineTransform2D3D << std::endl;


  
  // Load the perspective transformation
 
  if ( fileInputPerspective ) {

    typedef itk::PerspectiveProjectionTransform< double > PerspectiveProjectionTransformType;
    PerspectiveProjectionTransformType::Pointer perspectiveTransform = PerspectiveProjectionTransformType::New();

    itk::TransformFactory< PerspectiveProjectionTransformType >::RegisterTransform();

    typedef itk::TransformFileReader TransformFileReaderType;
    TransformFileReaderType::Pointer reader = TransformFileReaderType::New();
   
    reader->SetFileName( fileInputPerspective );
   
    try
      {
	reader->Update();
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << "Exception reading itk::PerspectiveProjectionTransformType< double, 3 >" << std::endl;
	std::cerr << err << std::endl;
	return EXIT_FAILURE;
      }
   
    typedef itk::TransformFileReader::TransformListType * TransformListType;
    TransformListType transforms = reader->GetTransformList();
   
    std::cout << "Number of transforms = " << transforms->size() << std::endl;
   
    itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();
   
    if( ! strcmp((*it)->GetNameOfClass(),"PerspectiveProjectionTransform")) {
     
      perspectiveTransform = static_cast< PerspectiveProjectionTransformType* >((*it).GetPointer());
    }
    else {
     
      std::cerr << "Failed to read transformation from: " << fileInputPerspective;
      return EXIT_FAILURE; 
    }
   
    perspectiveTransform->Print(std::cout);

    sid = perspectiveTransform->GetFocalDistance();
   
    double u, v;
    perspectiveTransform->GetOriginIn2D(u, v);
   
    o2Dx = (float) u;
    o2Dy = (float) v;
  }



  // The 'RayCastInterpolateImageFunction' is instantiated and passed the transform 
  // object. The 'RayCastInterpolateImageFunction' uses this
  // transform to reposition the x-ray source such that the DRR image
  // and x-ray source move as one around the input volume. This coupling
  // mimics the rigid geometry of the x-ray gantry. 

  typedef itk::RayCastInterpolateImageFunction<InputImageType,double> InterpolatorType;

  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  interpolator->SetTransform(affineTransform2D3D);

  // We can then specify a threshold above which the volume's
  // intensities will be integrated.

  interpolator->SetThreshold(threshold);

  // The ray-cast interpolator needs to know the initial position of the
  // ray source or focal point. In this example we place the input
  // volume at the origin and halfway between the ray source and the
  // screen. The distance between the ray source and the screen
  // is the "source to image distance" 'sid' and is specified by
  // the user. 

  InterpolatorType::InputPointType focalpoint;

  focalpoint[0] = origin3D[0] - translateVolume[0];
  focalpoint[1] = origin3D[1] - translateVolume[1];
  focalpoint[2] = origin3D[2] - (sid - translateVolume[2]);

  interpolator->SetFocalPoint(focalpoint);

  if (verbose)
    std::cout << "Focal Point: " 
	      << focalpoint[0] << ", " 
	      << focalpoint[1] << ", " 
	      << focalpoint[2] << endl;

  // Having initialised the interpolator we pass the object to the
  // resample filter.

  filter->SetInterpolator( interpolator );
  filter->SetTransform( affineTransform2D3D );

  // The size and resolution of the output DRR image is specified via the
  // resample filter. 

  // setup the scene
  InputImageType::SizeType   size;

  size[0] = dx;  // number of pixels along X of the 2D DRR image 
  size[1] = dy;  // number of pixels along Y of the 2D DRR image 
  size[2] = 1;   // only one slice

  filter->SetSize( size );

  double spacing[ Dimension ];

  spacing[0] = sx;  // pixel spacing along X of the 2D DRR image [mm]
  spacing[1] = sy;  // pixel spacing along Y of the 2D DRR image [mm]
  spacing[2] = 1.0; // slice thickness of the 2D DRR image [mm]

  filter->SetOutputSpacing( spacing );

  // In addition the position of the DRR is specified. The default
  // position of the input volume, prior to its transformation is
  // half-way between the ray source and screen and unless specified
  // otherwise the normal from the "screen" to the ray source passes
  // directly through the centre of the DRR.

  double origin2D[ Dimension ];

  origin2D[0] = origin3D[0] + o2Dx - sx*((double) dx - 1.)/2. + translateVolume[0]; 
  origin2D[1] = origin3D[1] + o2Dy - sy*((double) dy - 1.)/2. + translateVolume[1]; 
  origin2D[2] = origin3D[2] + translateVolume[2];

  filter->SetOutputOrigin( origin2D );

  if (verbose) 
    std::cout << "Detector size: " 
	      << size[0] << ", " 
	      << size[1] << ", " 
	      << size[2] << endl
	      << "  resolution: " 
	      << spacing[0] << ", " 
	      << spacing[1] << ", " 
	      << spacing[2] << endl
	      << "  position: " 
	      << origin2D[0] << ", " 
	      << origin2D[1] << ", " 
	      << origin2D[2] << endl;


  // Perform the projection
  // ~~~~~~~~~~~~~~~~~~~~~~

  filter->Update();


  // Write the projected image to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  OutputImageType::Pointer outputImage = OutputImageType::New();
 
  outputImage = filter->GetOutput();

  double myOrigin[] = {0, 0, 0}; // used to reset the origin of the DRR

  outputImage->SetOrigin(myOrigin);

  // create writer

  if (fileOutputDRR) 
    {
      // The output of the resample filter can then be passed to a writer to
      // save the DRR image to a file.

      typedef itk::ImageFileWriter< OutputImageType >  WriterType;
      WriterType::Pointer writer = WriterType::New();

      writer->SetFileName( fileOutputDRR );
      writer->SetInput( outputImage );

      try 
	{ 
	  std::cout << "Writing image: " << fileOutputDRR << endl;
	  writer->Update();
	} 
      catch( itk::ExceptionObject & err ) 
	{ 
	  std::cerr << "ERROR: ExceptionObject caught !" << std::endl; 
	  std::cerr << err << std::endl; 
	} 

    }
  else 
    {
      filter->Update();
    }

  
  // Save the affine transformation.
  // Used to transform the volume from the target space to the source.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputAffineTransformation ) {

    typedef itk::TransformFileWriter TransformFileWriterType;
    TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();

    itk::TransformFactory< AffineTransformType >::RegisterTransform();

    transformFileWriter->SetInput( affineTransform2D3D );
    transformFileWriter->SetFileName( fileOutputAffineTransformation ); 

    try
      {
	transformFileWriter->Update();         
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << "Exception when writing affine transformation" << std::endl;
	std::cerr << err << std::endl;
	return EXIT_FAILURE;
      }

    std::cout << "Affine transformation written to file: " 
	      << fileOutputAffineTransformation << std::endl;
  }


  // Save the inverse affine transformation with the origin at the
  // source. This will enable points in the 3D volume to be projected into 2D.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputAffineTransWithOriginAtSource ) {

    itk::Point<double, 3> pt1, pt2, pt3;

    typedef itk::EulerAffineTransform< double, 3 > EulerAffineTransformType;
  
    EulerAffineTransformType::Pointer eulerTransform3D    = EulerAffineTransformType::New();
    EulerAffineTransformType::Pointer invEulerTransform3D = EulerAffineTransformType::New();

    typedef itk::TransformFileWriter TransformFileWriterType;
    TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();

    itk::TransformFactory< EulerAffineTransformType >::RegisterTransform();

    // We set the EulerAffineTransform parameters to the
    // AffineTransform2D3D parameters (the translation, rotation
    // etc. order should match although the order of parameters differs).

    EulerAffineTransformType::ParametersType eulerParameters3D;
    EulerAffineTransformType::ParametersType invEulerParameters3D;
    AffineTransformType::ParametersType affineParameters3D = affineTransform2D3D->GetParameters();

    eulerParameters3D.SetSize( 12 );
    invEulerParameters3D.SetSize( 12 );

    double toDegrees = 180.0/vnl_math::pi;

    eulerParameters3D[ 0] = affineParameters3D[ 9] + origin3D[0]; // Translation
    eulerParameters3D[ 1] = affineParameters3D[10] + origin3D[1];
    eulerParameters3D[ 2] = affineParameters3D[11] + origin3D[2];
    
    eulerParameters3D[ 3] =  affineParameters3D[0]*toDegrees; // Rotation
    eulerParameters3D[ 4] = -affineParameters3D[1]*toDegrees;
    eulerParameters3D[ 5] =  affineParameters3D[2]*toDegrees;
    
    eulerParameters3D[ 6] = affineParameters3D[3]; // Scale
    eulerParameters3D[ 7] = affineParameters3D[4];
    eulerParameters3D[ 8] = affineParameters3D[5];
    
    eulerParameters3D[ 9] = affineParameters3D[6]; // Shear
    eulerParameters3D[10] = affineParameters3D[7];
    eulerParameters3D[11] = affineParameters3D[8];
      
    eulerTransform3D->SetParameters( eulerParameters3D );

    eulerTransform3D->Print( std::cout );

    // Now we calculate the inverse and shift the origin to the X-ray
    // source position

    eulerTransform3D->GetInverse( invEulerTransform3D );

    invEulerTransform3D->SetParametersFromTransform( invEulerTransform3D->GetFullAffineTransform() );

    invEulerParameters3D = invEulerTransform3D->GetParameters();

    invEulerParameters3D[0] += -translateVolume[0];
    invEulerParameters3D[1] += -translateVolume[1];
    invEulerParameters3D[2] += -translateVolume[2] + sid;

    invEulerTransform3D->SetParameters( invEulerParameters3D );

    invEulerTransform3D->Print( std::cout );

    transformFileWriter->SetInput( invEulerTransform3D );
    transformFileWriter->SetFileName( fileOutputAffineTransWithOriginAtSource ); 

    try
      {
	transformFileWriter->Update();         
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << "Exception when writing affine transformation" << std::endl;
	std::cerr << err << std::endl;
	return EXIT_FAILURE;
      }

    std::cout << "Affine transformation with origin at source written to file: " 
	      << fileOutputAffineTransWithOriginAtSource << std::endl;



  }

  
  // Save the perspective transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputPerspectiveTransformation ) {

    typedef itk::PerspectiveProjectionTransform< double > PerspectiveProjectionTransformType;
    PerspectiveProjectionTransformType::Pointer perspectiveTransform = PerspectiveProjectionTransformType::New();

    itk::TransformFactory< PerspectiveProjectionTransformType >::RegisterTransform();

    perspectiveTransform->SetFocalDistance( sid );
    perspectiveTransform->SetOriginIn2D( -o2Dx, -o2Dy );    

    typedef itk::TransformFileWriter TransformFileWriterType;
    TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();

    transformFileWriter->SetInput( perspectiveTransform );
    transformFileWriter->SetFileName( fileOutputPerspectiveTransformation ); 

    try
      {
	transformFileWriter->Update();         
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << "Exception when writing perspective transformation" << std::endl;
	std::cerr << err << std::endl;
	return EXIT_FAILURE;
      }

    std::cout << "Perspective transformation written to file: " 
	      << fileOutputPerspectiveTransformation << std::endl;
  }
 
  return 0;
}

