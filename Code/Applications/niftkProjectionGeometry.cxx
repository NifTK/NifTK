/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <ConversionUtils.h>
#include <CommandLineParser.h>

#include <itkGE5000_TomosynthesisGeometry.h>
#include <itkGE6000_TomosynthesisGeometry.h>
#include <itkIsocentricConeBeamRotationGeometry.h>

#include <itkTransformFileWriter.h>
#include <itkTransformFactory.h>
#include <itkNIFTKTransformIOFactory.h>

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL,   "Output verbose info"},
  {OPT_SWITCH, "dbg", NULL, "Output debugging info"},

  {OPT_INTx3|OPT_REQ,   "sz3D", "nx,ny,nz", "The size of the reconstructed volume in voxels"},
  {OPT_FLOATx3|OPT_REQ, "res3D", "rx,ry,rz", "The resolution of the reconstructed volume in mm"},

  {OPT_INTx2|OPT_REQ,   "sz2D", "nx,ny", "The size of the projection image in pixels"},
  {OPT_FLOATx2|OPT_REQ, "res2D", "rx,ry", "The resolution of the reconstructed volume in mm"},

  {OPT_INT,    "nProjs", "n",           "ISOCENTRIC: The number of projections [21]"},
  {OPT_DOUBLE, "FirstAngle", "theta",   "ISOCENTRIC: The angle of the first projection in the sequence [-89]"},
  {OPT_DOUBLE, "AngRange", "range",     "ISOCENTRIC: The full angular range of the sequence [180]"},
  {OPT_INT,    "axis", "number",        "ISOCENTRIC: The axis about which to rotate, 1:'x', 2:'y', 3:'z' [2:'y']"},
	    
  {OPT_DOUBLE,  "transX", "angle", "ISOCENTRIC: Add an additional translation in 'x' [none]"},
  {OPT_DOUBLE,  "transY", "angle", "ISOCENTRIC: Add an additional translation in 'y' [none]"},
  {OPT_DOUBLE,  "transZ", "angle", "ISOCENTRIC: Add an additional translation in 'z' [none]"},

  {OPT_DOUBLE, "FocalLength", "length", "The focal length of the projection [660]"},
  
  {OPT_SWITCH,  "GE5000", 0, "Use the 'old' GE-5000, 11 projection geometry [21 projection]"},
  {OPT_SWITCH,  "GE6000", 0, "Use the 'new' GE-6000, 15 projection geometry [21 projection]"},
  
  {OPT_DOUBLE,  "thetaX", "angle", "Add an additional rotation in 'x' [none]"},
  {OPT_DOUBLE,  "thetaY", "angle", "Add an additional rotation in 'y' [none]"},
  {OPT_DOUBLE,  "thetaZ", "angle", "Add an additional rotation in 'z' [none]"},

  {OPT_STRING|OPT_REQ,  "o", "filename", "Output file stem for the transformation files"},

  {OPT_DONE, NULL, NULL, 
   "Create a set of tomosynthesis projection matrices.\n"
  }
};

enum {
  O_VERBOSE = 0,
  O_DEBUG,

  O_RECONSTRUCTION_SIZE,
  O_RECONSTRUCTION_RES,

  O_PROJECTION_SIZE,
  O_PROJECTION_RES,

  O_NUMBER_OF_PROJECTIONS,
  O_FIRST_ANGLE,
  O_ANGULAR_RANGE,
  O_AXIS_NUMBER,

  O_TRANSX,
  O_TRANSY,
  O_TRANSZ,

  O_FOCAL_LENGTH,

  O_GE5000,
  O_GE6000,

  O_THETAX,
  O_THETAY,
  O_THETAZ,

  O_OUTPUT_FILESTEM
};
 

/**
 * \brief Create a set of tomosynthesis projection matrices.
 */
int main(int argc, char** argv)
{
  std::string fileOutputFilestem;

  bool flgGE_5000 = false;	// Use the GE 5000 11 projection geometry
  bool flgGE_6000 = false;	// Use the GE 6000 15 projection geometry

  bool flgFirstAngleSet = false; // Has the user set the first angle

  bool flgTransX = false;	// Translation in 'x' has been set
  bool flgTransY = false;	// Translation in 'y' has been set
  bool flgTransZ = false;	// Translation in 'z' has been set

  unsigned int nProjections = 0; // The number of projections in the sequence

  int axis  = 0;		// The axis about which to rotate

  int *clo_size = 0;		// The size of the reconstructed volume
  int *proj_size = 0;		// The size of the projection

  float *clo_res = 0;		// The resolution of the reconstructed volume
  float *proj_res = 0;		// The resolution of the projection

  double firstAngle = 0;         // The angle of the first projection in the sequence
  double angularRange = 0;       // The full angular range of the sequence
  double focalLength = 0;        // The focal length of the projection

  double thetaX = 0;		 // An additional rotation in 'x'
  double thetaY = 0;		 // An additional rotation in 'y'
  double thetaZ = 0;		 // An additional rotation in 'z'

  double transX = 0;		 // An additional translation in 'x'
  double transY = 0;		 // An additional translation in 'y'
  double transZ = 0;		 // An additional translation in 'z'

  typedef float IntensityType;
  typedef itk::ProjectionGeometry< IntensityType > ProjectionGeometryType;

  ProjectionGeometryType::ProjectionSizeType pProjectionSize;
  pProjectionSize.Fill(0);

  ProjectionGeometryType::ProjectionSpacingType pProjectionSpacing(0.0);

  ProjectionGeometryType::VolumeSizeType pVolumeSize;
  pVolumeSize.Fill(0);

  ProjectionGeometryType::VolumeSpacingType pVolumeSpacing(0.0);
  
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);
  
  if (CommandLineOptions.GetArgument(O_RECONSTRUCTION_SIZE, clo_size)) {
    pVolumeSize[0] = clo_size[0];
    pVolumeSize[1] = clo_size[1];
    pVolumeSize[2] = clo_size[2];
  }

  if (CommandLineOptions.GetArgument(O_RECONSTRUCTION_RES, clo_res)) {
    pVolumeSpacing[0] = clo_res[0];
    pVolumeSpacing[1] = clo_res[1];
    pVolumeSpacing[2] = clo_res[2];
  }

  if (CommandLineOptions.GetArgument(O_PROJECTION_SIZE, proj_size)) {
    pProjectionSize[0] = proj_size[0];
    pProjectionSize[1] = proj_size[1];
  }

  if (CommandLineOptions.GetArgument(O_PROJECTION_RES, proj_res)) {
    pProjectionSpacing[0] = proj_res[0];
    pProjectionSpacing[1] = proj_res[1];
  }

  CommandLineOptions.GetArgument(O_NUMBER_OF_PROJECTIONS, nProjections);
  flgFirstAngleSet = CommandLineOptions.GetArgument(O_FIRST_ANGLE, firstAngle);
  CommandLineOptions.GetArgument(O_ANGULAR_RANGE, angularRange);
  CommandLineOptions.GetArgument(O_AXIS_NUMBER, axis);

  flgTransX = CommandLineOptions.GetArgument(O_TRANSX, transX);
  flgTransY = CommandLineOptions.GetArgument(O_TRANSY, transY);
  flgTransZ = CommandLineOptions.GetArgument(O_TRANSZ, transZ);

  CommandLineOptions.GetArgument(O_FOCAL_LENGTH, focalLength);

  CommandLineOptions.GetArgument(O_GE5000, flgGE_5000);
  CommandLineOptions.GetArgument(O_GE6000, flgGE_6000);

  CommandLineOptions.GetArgument(O_THETAX, thetaX);
  CommandLineOptions.GetArgument(O_THETAY, thetaY);
  CommandLineOptions.GetArgument(O_THETAZ, thetaZ);

  CommandLineOptions.GetArgument(O_OUTPUT_FILESTEM, fileOutputFilestem);


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputFilestem.length() == 0 ) {
    CommandLineOptions.PrintUsage();
    return EXIT_FAILURE;
  }

  

  if ( flgGE_5000 && flgGE_6000 ) {
    std::cout << "Command line options '-GE5000' and '-GE6000' are exclusive." << std::endl;

    CommandLineOptions.PrintUsage();
    return EXIT_FAILURE;
  }
       
  if ( (flgGE_5000 || flgGE_6000) && (nProjections || flgFirstAngleSet || angularRange || focalLength) ) {
    std::cout << "Command line options '-GE5000' or '-GE6000' and '-nProjs' "
				   "or '-1stAngle' or '-AngRange' or '-FocalLength' are exclusive." << std::endl;

    CommandLineOptions.PrintUsage();
    return EXIT_FAILURE;
  }
      
  if ( (flgGE_5000 || flgGE_6000) && (flgTransX || flgTransY || flgTransZ) ) {
    std::cout << "Command line options '-transX|Y|Z' can only be used with isocentric geometry." << std::endl;
    return EXIT_FAILURE;
  }



  // Create the tomosynthesis geometry
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  char filename[256];

  ProjectionGeometryType::Pointer geometry; 

  // Create the GE-5000 11 projection geometry 

  if (flgGE_5000) {

    typedef itk::GE5000_TomosynthesisGeometry< IntensityType > GE5000_TomosynthesisGeometryType;
    geometry = GE5000_TomosynthesisGeometryType::New();
  }

  // Create the GE-6000 15 projection geometry 

  else if (flgGE_6000) {

    typedef itk::GE6000_TomosynthesisGeometry< IntensityType > GE6000_TomosynthesisGeometryType;
    geometry = GE6000_TomosynthesisGeometryType::New();
  }

  // Create an isocentric cone bean rotation geometry

  else {
  
    if (! nProjections) nProjections = 21;
    if (! flgFirstAngleSet) firstAngle = -89.;
    if (! angularRange) angularRange = 180.;
    if (! focalLength) focalLength = 660.;

    typedef itk::IsocentricConeBeamRotationGeometry< IntensityType > IsocentricConeBeamRotationGeometryType;

    IsocentricConeBeamRotationGeometryType::Pointer isoGeometry = IsocentricConeBeamRotationGeometryType::New();

    isoGeometry->SetNumberOfProjections(nProjections);
    isoGeometry->SetFirstAngle(firstAngle);
    isoGeometry->SetAngularRange(angularRange);
    isoGeometry->SetFocalLength(focalLength);

    isoGeometry->SetTranslation(transX, transY, transZ);

    if (axis) {

      switch (axis) 
	{

	case 1: {
	  isoGeometry->SetRotationAxis(itk::ISOCENTRIC_CONE_BEAM_ROTATION_IN_X);
	  break;
	}

	case 2: {
	  isoGeometry->SetRotationAxis(itk::ISOCENTRIC_CONE_BEAM_ROTATION_IN_Y);
	  break;
	}

	case 3: {
	  isoGeometry->SetRotationAxis(itk::ISOCENTRIC_CONE_BEAM_ROTATION_IN_Z);
	  break;
	}

	default: {
	  std::cout << "Command line option '-axis' must be: 1, 2 or 3." << std::endl;
	  
	  CommandLineOptions.PrintUsage();
	  return EXIT_FAILURE;
	}
	}
    }

    geometry = isoGeometry;
  }

  // Specify the projection and volume sizes

  if (thetaX) geometry->SetRotationInX(thetaX);
  if (thetaY) geometry->SetRotationInY(thetaY);
  if (thetaZ) geometry->SetRotationInZ(thetaZ);

  geometry->SetProjectionSize(pProjectionSize);
  geometry->SetProjectionSpacing(pProjectionSpacing);

  geometry->SetVolumeSize(pVolumeSize);
  geometry->SetVolumeSpacing(pVolumeSpacing);

  geometry->Print( std::cout );

  
  // Output the projection and affine transformations
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  itk::ObjectFactoryBase::RegisterFactory(itk::NIFTKTransformIOFactory::New());

  ProjectionGeometryType::EulerAffineTransformPointerType pAffineTransform;
  ProjectionGeometryType::PerspectiveProjectionTransformPointerType pPerspectiveTransform;

  itk::TransformFactory< ProjectionGeometryType::PerspectiveProjectionTransformType >::RegisterTransform();
  itk::TransformFactory< ProjectionGeometryType::EulerAffineTransformType >::RegisterTransform();

  typedef itk::TransformFileWriter TransformFileWriterType;
  TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();

  unsigned int iProjection;

  for (iProjection=0; iProjection<geometry->GetNumberOfProjections(); iProjection++) {

    // Get and write the affine transform

    try {
      pAffineTransform = geometry->GetAffineTransform(iProjection);
    }

    catch( itk::ExceptionObject & err ) { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }                

    if (nProjections < 100) 
      sprintf(filename, "%s_%02d.tAffine", fileOutputFilestem.c_str(), iProjection);
    else
      sprintf(filename, "%s_%03d.tAffine", fileOutputFilestem.c_str(), iProjection);

    transformFileWriter->SetInput(pAffineTransform);
    transformFileWriter->SetFileName(std::string(filename)); 
    transformFileWriter->Update();         
    
    std::cout << "Writing affine transform: " << filename << std::endl;
  

    // Get and write the perspective transform

    try {
      pPerspectiveTransform = geometry->GetPerspectiveTransform(iProjection);
    }

    catch( itk::ExceptionObject & err ) { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }                


    if (nProjections < 100) 
      sprintf(filename, "%s_%02d.tPerspective", fileOutputFilestem.c_str(), iProjection);
    else  
      sprintf(filename, "%s_%03d.tPerspective", fileOutputFilestem.c_str(), iProjection);

    transformFileWriter->SetInput(pPerspectiveTransform);
    transformFileWriter->SetFileName(std::string(filename)); 
    transformFileWriter->Update();         

    std::cout << "Writing perspective transform: " << filename << std::endl;
  }


  std::cout << "Done" << std::endl;
  
  return EXIT_SUCCESS;   
}


