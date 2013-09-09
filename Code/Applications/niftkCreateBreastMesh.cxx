/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkConversionUtils.h>
#include <niftkCommandLineParser.h>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkImageToVTKImageFilter.h>
#include <itkConnectedThresholdImageFilter.h>
#include <itkCurvatureFlowImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkInvertIntensityImageFilter.h>
#include <itkResampleImageFilter.h>
#include <itkAffineTransform.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkThresholdImageWithRespectToPlane.h>
#include <itkSetBoundaryVoxelsToValueFilter.h>
#include <itkSignedMaurerDistanceMapImageFilter.h>

#include <vtkCamera.h>
#include <vtkDecimatePro.h>
#include <vtkQuadricDecimation.h>
#include <vtkFollower.h> 
#include <vtkGeometryFilter.h> 
#include <vtkImageShrink3D.h>
#include <vtkInteractorStyleTrackballCamera.h> 
#include <vtkLODActor.h> 
#include <vtkLineSource.h> 
#include <vtkMarchingCubes.h> 
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h> 
#include <vtkPolyDataNormals.h> 
#include <vtkPolyDataReader.h> 
#include <vtkPolyDataWriter.h> 
#include <vtkSTLWriter.h> 
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTriangleFilter.h>
#include <vtkTubeFilter.h> 
#include <vtkVectorText.h> 
#include <vtkWindowedSincPolyDataFilter.h> 
#include <vtkFloatArray.h>
#include <vtkCellArray.h>


struct niftk::CommandLineArgumentDescription clArgList[] = {

   {OPT_SWITCH, "dbg", 0, "Output debugging information."},
   {OPT_SWITCH, "v", 0, "Verbose output during execution."},

   {OPT_DOUBLE, "res", "resolution", "The isotropic volume resolution in mm for sub-sampling [5]."},

   {OPT_INTx3, "seed", "i,j,k", "Voxel index used to seed the background region growing [0,0,0]."},

   {OPT_DOUBLEx3, "pp", "px,py,pz", "The position of the plane in mm to threshold the image with."},
   {OPT_DOUBLEx3, "pn", "nx,ny,nz", "The normal of the plane to threshold the image with."},

   {OPT_FLOAT, "dt", "distance", 
    "The distance transform threshold in voxels^2 (negative = external, positive = internal, eg. -2.) [none]."},

   // The following are all parameters to control the VTK marching cubes algorithm and related processing

   {OPT_SWITCH, "ni", 0, "Do not start the interactor.\n"},

   {OPT_SWITCH, "s1", 0, "Smooth the poly data BEFORE decimation."},
   {OPT_SWITCH, "s2", 0, "Smooth the poly data AFTER decimation."},

   {OPT_INTx3, "shrink", "sx,sy,sz",
    "Sub-sample the input image on a uniform grid."},

   {OPT_INT, "minPD", "nCells",
    "The minimum number of cells that the extracted poly data should have."},

   {OPT_INT, "si", "nIterations",
    "The number of smoothing iterations to apply (0-200) [20]."},
   {OPT_FLOAT, "sw", "bandwidth",
    "The band width of the smoothing filter (0-2) [0.1]."},

   {OPT_SWITCH, "dpro", 0, "Use vtkDecimatePro decimation [default: vtkQuadricDecimation]."},
   {OPT_FLOAT, "dc", "factor", 
    "The decimation factor (0-1) eg. 0.9 = 90%."},

   {OPT_FLOAT, "fa", "angle",
    "The angle definition of an edge (0-90) decimation and/or smoothing."},
   
   {OPT_FLOAT, "t", "threshold",
    "The marching cubes isosurface value [mid-range]."},

   // IO Options

   {OPT_STRING, "oDT", "fileDistTransOut", "Filename to write the distance transform to."},
   {OPT_STRING, "oMask", "fileMaskOut", "Filename of the output mask image."},
   {OPT_STRING, "oPlane", "filePlaneOut", "Filename to write the mask thresholded by a plane to."},

   {OPT_SWITCH, "text", 0, "Output the vtk surface in text format [binary]."},
   {OPT_STRING, "oSTL", "fileSTLout", "Filename of the output STL (stereo lithography) surface."},
   {OPT_STRING, "oSMESH", "fileSMESHout", "Filename of the output SMESH surface."},
   {OPT_STRING, "o", "fileVTKout", "Filename of the output vtk surface."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "The input image."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to create a VTK polydata mesh from a breast MRI image.\n"
  }
};

enum {
  O_DEBUG = 0,
  O_VERBOSE,

  O_SUBSAMPLING_RESOLUTION,

  O_REGION_GROWING_SEED_INDEX,

  O_PLANE_POSITION,
  O_PLANE_NORMAL,

  O_DISTANCE_TRANSFORM_THRESHOLD,

  O_NOINTERACTOR,

  O_PRESMOOTH,
  O_POSTSMOOTH,

  O_SHRINK,

  O_MIN_NO_CELLS,

  O_NITERATIONS,
  O_BANDWIDTH,

  O_USE_VTK_DECIMATE_PRO,
  O_DECIMATION,

  O_FEATUREANGLE,

  O_ISOSURFACE,

  O_DISTTRANS_OUTPUT_FILE,
  O_MASK_OUTPUT_FILE,
  O_PLANE_OUTPUT_FILE,

  O_TEXT_OUTPUT,
  O_STL_OUTPUT_FILE,
  O_SMESH_OUTPUT_FILE,
  O_VTK_OUTPUT_FILE,

  O_INPUT_IMAGE
};


void polyDataInfo(vtkPolyData *polyData);
void WritePolydataAsTetgenSmeshFile( const char *smesh_file, vtkPolyData *surface );


int main (int argc, char *argv[])
{
  std::string fileInputImage;	// Input voxel image

  std::string fileOutputDistanceTransform; // Output distance transform image
  std::string fileOutputPlane;	  // Output image thresholded by a plane
  std::string fileOutputMask;	  // Output mask image
  std::string fileOutputPolydata; // Output polydata VTK surface file
  std::string fileOutputSTL;    // Output polydata STL surface file
  std::string fileOutputSMESH;  // Output SMESH surface file

  bool debug;			// Output debugging information
  bool verbose;			// Verbose output during execution

  bool noInteractor;		// Don't start the interactor

  bool preSmooth;		// Smooth the poly data BEFORE decimation.
  bool postSmooth;		// Smooth the poly data AFTER decimation.

  bool flgDecimatePro;		// Use the vtkDecimatePro filter [default: vtkQuadricDecimation]

  int flgTextOutput = 0;	// Output the vtk surface in text format

  int minNCells = 0;		// The minimum no. of cells permissible

  int niterations = 20;		// The number of smoothing iterations

  int *shrink = 0;		// Sub-sample the input image

  int *seedIndex = 0;		// The region growing seed index

  float bandwidth = 0.1;	// The band width of the smoothing filter

  float decimation = 0.;	// The decimation factor to be applied

  float featureAngle = 0.;	// The decimation feature angle

  float isoSurface = 0;		// The marching cubes isosurface value

  float imMaximum;		// The minimum intensity in the input image
  float imMinimum;		// The maximum intensity in the input image

  float distTransThresh = 0;  // The distance transform threshold in voxels^2 
                                // (negative = external, positive = internal)

  double subsamplingResolution = 5.; //The isotropic volume resolution in mm for sub-sampling

  double *planePosition = 0;    // The position of the plane in mm to threshold the image with."},
  double *planeNormal = 0;      // The normal of the plane to threshold the image with."},

				// The link between objects in the pipeline
  vtkImageData *pipeVTKImageDataConnector;
  vtkPolyData *pipeVTKPolyDataConnector;	// The link between objects in the pipeline


  // Define the input image type
  const unsigned int ImageDimension = 3;

  typedef float InputPixelType;
  typedef itk::Image<InputPixelType, ImageDimension> InputImageType;

  InputImageType::Pointer pipeITKImageDataConnector;

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);


  CommandLineOptions.GetArgument(O_DEBUG, debug);
  CommandLineOptions.GetArgument(O_VERBOSE, verbose);

  CommandLineOptions.GetArgument(O_SUBSAMPLING_RESOLUTION, subsamplingResolution);

  CommandLineOptions.GetArgument(O_NOINTERACTOR, noInteractor);

  CommandLineOptions.GetArgument(O_REGION_GROWING_SEED_INDEX, seedIndex);
  
  CommandLineOptions.GetArgument(O_PLANE_POSITION, planePosition);
  CommandLineOptions.GetArgument(O_PLANE_NORMAL,   planeNormal);


  CommandLineOptions.GetArgument(O_PRESMOOTH, preSmooth);
  CommandLineOptions.GetArgument(O_POSTSMOOTH, postSmooth);

  CommandLineOptions.GetArgument(O_DISTANCE_TRANSFORM_THRESHOLD, distTransThresh);

  CommandLineOptions.GetArgument(O_SHRINK, shrink);

  CommandLineOptions.GetArgument(O_MIN_NO_CELLS, minNCells);
  

  if (CommandLineOptions.GetArgument(O_NITERATIONS, niterations) && (niterations > 0)
      && (! (preSmooth || postSmooth))) {

    std::cout << "No. of smoothing iterations specified ("
                                  << niftk::ConvertToString(niterations)
                                  << ") post-decimation smoothing assumed."<< std::endl;
    postSmooth = true;
  }

  if ((niterations < 0) || (niterations > 200)) {
    std::cerr <<"No. of smoothing iterations ("
                                     + niftk::ConvertToString(niterations)
                                     + " is outside allowed range (0-200).";
    return EXIT_FAILURE;
  }

  if (CommandLineOptions.GetArgument(O_BANDWIDTH, bandwidth) 
      && (! (preSmooth || postSmooth))) {
    std::cout << "WARNING: Smoothing bandwith specified post-decimation smoothing assumed."<< std::endl;
    postSmooth = 1;
  }

  if ((bandwidth < 0.) || (bandwidth > 2)) {
    std::cerr <<"Smoothing bandwith (" << niftk::ConvertToString(bandwidth)
                  << ") is outside allowed range (0-2).";
    return EXIT_FAILURE;
  }

  CommandLineOptions.GetArgument(O_POSTSMOOTH, postSmooth);
  
  if (CommandLineOptions.GetArgument(O_USE_VTK_DECIMATE_PRO, flgDecimatePro) && verbose)
    std::cout << "Using 'vtkDecimatePro' decimation [default: vtkQuadricDecimation]"<< std::endl;

  if (CommandLineOptions.GetArgument(O_DECIMATION, decimation) && decimation && verbose)
    std::cout << "Decimation factor: " << niftk::ConvertToString(decimation) + " applied"<< std::endl;

  if (decimation && ((decimation < 0.) || (decimation > 1.))) {
    std::cerr << "Decimation value (" << niftk::ConvertToString(decimation)
                << ") is outside allowed range (0-1)";
    return EXIT_FAILURE;
  }

  if (CommandLineOptions.GetArgument(O_FEATUREANGLE, featureAngle) && verbose)
    std::cout << "Decimation feature angle: " << niftk::ConvertToString(featureAngle)<< std::endl;

  if (featureAngle && ((featureAngle < 0.) || (featureAngle > 90.))) {
    std::cerr <<"Decimation feature angle value (" << niftk::ConvertToString(featureAngle)
                                   << ") is outside allowed range (0-90)";
    return EXIT_FAILURE;
  }

  CommandLineOptions.GetArgument(O_ISOSURFACE, isoSurface);

  CommandLineOptions.GetArgument(O_DISTTRANS_OUTPUT_FILE, fileOutputDistanceTransform);
  CommandLineOptions.GetArgument(O_PLANE_OUTPUT_FILE, fileOutputPlane);
  CommandLineOptions.GetArgument(O_MASK_OUTPUT_FILE, fileOutputMask);
  CommandLineOptions.GetArgument(O_VTK_OUTPUT_FILE, fileOutputPolydata);
  CommandLineOptions.GetArgument(O_STL_OUTPUT_FILE, fileOutputSTL);
  CommandLineOptions.GetArgument(O_SMESH_OUTPUT_FILE, fileOutputSMESH);

  CommandLineOptions.GetArgument( O_INPUT_IMAGE, fileInputImage );


  if (CommandLineOptions.GetArgument(O_TEXT_OUTPUT, flgTextOutput) && verbose)
      std::cout << "Output vtk surface will be in TEXT format"<< std::endl;
  else
      std::cout << "Output vtk surface will be in BINARY format"<< std::endl;


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  typedef itk::ImageFileReader< InputImageType > FileReaderType;

  FileReaderType::Pointer imageReader = FileReaderType::New();

  imageReader->SetFileName(fileInputImage);

  try
  { 
    std::cout << "Reading the input image..."<< std::endl;
    imageReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }

  InputImageType::Pointer inputImage = imageReader->GetOutput();

  const InputImageType::SpacingType& sp = inputImage->GetSpacing();
  std::cout << "Input image resolution: "
				<< niftk::ConvertToString(sp[0]) << ","
				<< niftk::ConvertToString(sp[1]) << ","
				<< niftk::ConvertToString(sp[2])<< std::endl;

  const InputImageType::SizeType& sz = inputImage->GetLargestPossibleRegion().GetSize();
  std::cout << "Input image dimensions: "
				<< niftk::ConvertToString(sz[0]) << ","
				<< niftk::ConvertToString(sz[1]) << ","
				<< niftk::ConvertToString(sz[2])<< std::endl;
  

  // Get max and min intensities
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::MinimumMaximumImageCalculator< InputImageType > MinimumMaximumImageCalculatorType;
  
  MinimumMaximumImageCalculatorType::Pointer  rangeCalculator= MinimumMaximumImageCalculatorType::New();
  
  rangeCalculator->SetImage( imageReader->GetOutput() );
  rangeCalculator->Compute();

  imMaximum = rangeCalculator->GetMaximum();
  imMinimum = rangeCalculator->GetMinimum();
  
  std::cout << "Input image intensity range: "
				<< niftk::ConvertToString(imMinimum) << " to "
				<< niftk::ConvertToString(imMaximum)<< std::endl;
  
  if (! isoSurface ) 
      isoSurface = ( imMaximum - imMinimum ) / 2.;


  // Region grow the background as we're not interested in internal
  // breast structure.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  //  The smoothing filter is instantiated using the image type as
  //  a template parameter.

  typedef itk::CurvatureFlowImageFilter< InputImageType, InputImageType > CurvatureFlowImageFilterType;

  CurvatureFlowImageFilterType::Pointer preRegionGrowingSmoothing = CurvatureFlowImageFilterType::New();

  preRegionGrowingSmoothing->SetInput( imageReader->GetOutput() );

  unsigned int nItersPreRegionGrowingSmoothing = 5;
  double timeStepPreRegionGrowingSmoothing = 0.125;

  preRegionGrowingSmoothing->SetNumberOfIterations( nItersPreRegionGrowingSmoothing );
  preRegionGrowingSmoothing->SetTimeStep( 0.125 );

  try
  { 
    std::cout << "Applying pre-region-growing smoothing, no. iters: "
                                  << niftk::ConvertToString(nItersPreRegionGrowingSmoothing)
                                  << ", time step: "
                                  << niftk::ConvertToString(timeStepPreRegionGrowingSmoothing)
                                  << "..."<< std::endl;
    preRegionGrowingSmoothing->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }


  //  We now declare the type of the region growing filter. In this case it is
  //  the ConnectedThresholdImageFilter. 

  typedef itk::ConnectedThresholdImageFilter< InputImageType, InputImageType > ConnectedFilterType;

  ConnectedFilterType::Pointer connectedThreshold = ConnectedFilterType::New();

  connectedThreshold->SetInput( preRegionGrowingSmoothing->GetOutput() );

  connectedThreshold->SetLower( imMinimum  );
  connectedThreshold->SetUpper( isoSurface );

  connectedThreshold->SetReplaceValue( 1000 );

  InputImageType::IndexType  index;
  
  if ( seedIndex ) {
      
      index[0] = seedIndex[0];
      index[1] = seedIndex[1];
      index[2] = seedIndex[2];
  }
  else {
      
      index[0] = 0;
      index[1] = 0;
      index[2] = 0;
  }

  connectedThreshold->SetSeed( index );

  try
  { 
    std::cout << "Region-growing the image background between: "
                                  << niftk::ConvertToString(imMinimum) << " and "
                                  << niftk::ConvertToString(isoSurface) << "..."<< std::endl;
    connectedThreshold->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }


  // Invert the segmentation

#if 0
  typedef itk::InvertIntensityImageFilter< InputImageType, InputImageType > InvertIntensityImageFilterType;

  InvertIntensityImageFilterType::Pointer invertSegmentation = InvertIntensityImageFilterType::New();

  invertSegmentation->SetInput( connectedThreshold->GetOutput() );
#endif

  // Write the segmentation to a file

  if ( fileOutputMask.length() > 0 ) {

      typedef unsigned char OutputMaskPixelType;
      typedef itk::Image< OutputMaskPixelType, ImageDimension > OutputMaskImageType;

      typedef itk::RescaleIntensityImageFilter< InputImageType, OutputMaskImageType > RescaleMaskFilterType;
      RescaleMaskFilterType::Pointer casterMask = RescaleMaskFilterType::New();

      casterMask->SetInput( connectedThreshold->GetOutput() );
      casterMask->SetOutputMaximum( 1 );
                        
      typedef  itk::ImageFileWriter<  OutputMaskImageType  > MaskWriterType;
      MaskWriterType::Pointer writerMask = MaskWriterType::New();
      
      writerMask->SetFileName( fileOutputMask );
      writerMask->SetInput( casterMask->GetOutput() );

      try {
        std::cout << "Writing the segmentation to file: " << fileOutputMask.c_str()<< std::endl;
        writerMask->Update();
      }
      catch( itk::ExceptionObject & excep ) {
        std::cerr << "Exception caught !" << std::endl;
        std::cerr << excep << std::endl;
      }
  }

  // Write the mask thresholded by the input plane

  
  if (( planePosition || planeNormal ) && ( fileOutputPlane.length() > 0 )) {
  
    typedef unsigned char OutputMaskPixelType;
    typedef itk::Image< OutputMaskPixelType, ImageDimension > OutputMaskImageType;

    typedef itk::RescaleIntensityImageFilter< InputImageType, OutputMaskImageType > RescaleMaskFilterType;
    RescaleMaskFilterType::Pointer casterMask = RescaleMaskFilterType::New();

    casterMask->SetInput( connectedThreshold->GetOutput() );
    casterMask->SetOutputMaximum( 1 );

    typedef itk::ThresholdImageWithRespectToPlane < OutputMaskImageType, OutputMaskImageType > ThresholdImageWithRespectToPlaneType;
  
    ThresholdImageWithRespectToPlaneType::Pointer thresholdWithRespectToPlane = ThresholdImageWithRespectToPlaneType::New();

    if ( planePosition )
      thresholdWithRespectToPlane->SetPlanePosition( planePosition[0], planePosition[1], planePosition[2] );
    
    if ( planeNormal )
      thresholdWithRespectToPlane->SetPlaneNormal( planeNormal[0], planeNormal[1], planeNormal[2] );

    thresholdWithRespectToPlane->SetInput( casterMask->GetOutput() );

    thresholdWithRespectToPlane->SetThresholdValue( 1 );

    try
    { 
      std::cout << "Thresholding mask voxels on one side of a plane..."<< std::endl;
      thresholdWithRespectToPlane->Update();
    }
    catch (itk::ExceptionObject &ex)
    { 
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
    }

    typedef  itk::ImageFileWriter< OutputMaskImageType > WriterType;
    WriterType::Pointer writer = WriterType::New();
    
    writer->SetFileName( fileOutputPlane );
    writer->SetInput( thresholdWithRespectToPlane->GetOutput() );
    
    try {
      std::cout << "Writing mask thresholded by plane to file: " << fileOutputPlane.c_str()<< std::endl;
      writer->Update();
    }
    catch( itk::ExceptionObject & excep ) {
      std::cerr << "Exception caught !" << std::endl;
      std::cerr << excep << std::endl;
    }
  }



  // Blur the region grown segmentation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::DiscreteGaussianImageFilter< InputImageType, InputImageType > DiscreteGaussianImageFilterType;

  DiscreteGaussianImageFilterType::Pointer blurSegmentation = DiscreteGaussianImageFilterType::New();

  blurSegmentation->SetInput( connectedThreshold->GetOutput() );

  blurSegmentation->SetUseImageSpacingOn();
  blurSegmentation->SetVariance( 1.0 );

  try
  { 
    std::cout << "Blurring the region-grown segmentation..."<< std::endl;
    blurSegmentation->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }


  // Downsample the image to istropic voxels with dimensions
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ResampleImageFilter< InputImageType, InputImageType > ResampleImageFilterType;
  ResampleImageFilterType::Pointer subsampleFilter = ResampleImageFilterType::New();
  
  subsampleFilter->SetInput( blurSegmentation->GetOutput() );

  double spacing[ ImageDimension ];
  spacing[0] = subsamplingResolution; // pixel spacing in millimeters along X
  spacing[1] = subsamplingResolution; // pixel spacing in millimeters along Y
  spacing[2] = subsamplingResolution; // pixel spacing in millimeters along Z

  subsampleFilter->SetOutputSpacing( spacing );

  double origin[ ImageDimension ];
  origin[0] = 0.0;  // X space coordinate of origin
  origin[1] = 0.0;  // Y space coordinate of origin
  origin[2] = 0.0;  // Y space coordinate of origin

  subsampleFilter->SetOutputOrigin( origin );

  InputImageType::DirectionType direction;
  direction.SetIdentity();
  subsampleFilter->SetOutputDirection( direction );

  InputImageType::SizeType   size;

  size[0] = (int) ceil( sz[0]*sp[0]/spacing[0] );  // number of pixels along X
  size[1] = (int) ceil( sz[1]*sp[1]/spacing[1] );  // number of pixels along X
  size[2] = (int) ceil( sz[2]*sp[2]/spacing[2] );  // number of pixels along X

  subsampleFilter->SetSize( size );

  typedef itk::AffineTransform< double, ImageDimension >  TransformType;
  TransformType::Pointer transform = TransformType::New();

  subsampleFilter->SetTransform( transform );

  typedef itk::LinearInterpolateImageFunction< InputImageType, double >  InterpolatorType;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();
 
  subsampleFilter->SetInterpolator( interpolator );

  subsampleFilter->SetDefaultPixelValue( 0 );

  try
  { 
    std::cout << "Resampling image to dimensions: "
                                  << niftk::ConvertToString(size[0]) << ", "
                                  << niftk::ConvertToString(size[1]) << ", "
                                  << niftk::ConvertToString(size[2])
                                  << "voxels, with resolution : "
                                  << niftk::ConvertToString(spacing[0]) << ", "
                                  << niftk::ConvertToString(spacing[1]) << ", "
                                  << niftk::ConvertToString(spacing[2]) << "mm..."<< std::endl;

    subsampleFilter->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }


  // Compute a distance transform to smooth the boundary and enable troughs to be filled
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( distTransThresh ) {

    typedef itk::SignedMaurerDistanceMapImageFilter< InputImageType, InputImageType > SignedMaurerDistanceMapImageFilterType;

    SignedMaurerDistanceMapImageFilterType::Pointer distanceTransform = SignedMaurerDistanceMapImageFilterType::New();

    distanceTransform->SetInput( subsampleFilter->GetOutput() );
  
    try
    { 
      std::cout << "Computing distance transform..."<< std::endl;
      distanceTransform->Update();
    }
    catch (itk::ExceptionObject &ex)
    { 
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
    }

    if ( fileOutputDistanceTransform.length() > 0 ) {
      
      typedef  itk::ImageFileWriter<  InputImageType  > WriterType;
      WriterType::Pointer writer = WriterType::New();
      
      writer->SetFileName( fileOutputDistanceTransform );
      writer->SetInput( distanceTransform->GetOutput() );
      
      try {
        std::cout << "Writing distance transform to file: "
                                      << fileOutputDistanceTransform.c_str()<< std::endl;
        writer->Update();
      }
      catch( itk::ExceptionObject & excep ) {
        std::cerr << "Exception caught !" << std::endl;
        std::cerr << excep << std::endl;
      }
    }

    pipeITKImageDataConnector = distanceTransform->GetOutput();
  }
  else
    pipeITKImageDataConnector = subsampleFilter->GetOutput();


  // Set the border around the image to zero to prevent holes in the image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 
  typedef itk::SetBoundaryVoxelsToValueFilter< InputImageType, InputImageType > SetBoundaryVoxelsToValueFilterType;

  SetBoundaryVoxelsToValueFilterType::Pointer setBoundary = SetBoundaryVoxelsToValueFilterType::New();

  setBoundary->SetInput( pipeITKImageDataConnector );

  if ( distTransThresh ) 
    // The threshold '-1000' corresponds to voxels which are a long way from the region grown boundary
    setBoundary->SetValue( -1000 );
  else
    // This is the region growing value
    setBoundary->SetValue( 1000 );

  try
  { 
    std::cout << "Sealing the image boundary..."<< std::endl;
    setBoundary->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }
 

  // Threshold voxels on one side of a plane
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( planePosition || planeNormal ) {
  
    typedef itk::ThresholdImageWithRespectToPlane < InputImageType, InputImageType > ThresholdImageWithRespectToPlaneType;
  
    ThresholdImageWithRespectToPlaneType::Pointer thresholdWithRespectToPlane = ThresholdImageWithRespectToPlaneType::New();

    if ( planePosition )
      thresholdWithRespectToPlane->SetPlanePosition( planePosition[0], planePosition[1], planePosition[2] );
    
    if ( planeNormal )
      thresholdWithRespectToPlane->SetPlaneNormal( planeNormal[0], planeNormal[1], planeNormal[2] );

    thresholdWithRespectToPlane->SetInput(  setBoundary->GetOutput() );

    if ( distTransThresh ) 
      // The threshold '-1000' corresponds to voxels which are a long way from the region grown boundary
      thresholdWithRespectToPlane->SetThresholdValue( -1000 );
    else
      // This is the region growing value
      thresholdWithRespectToPlane->SetThresholdValue( 1000 );

    try
    { 
      std::cout << "Thresholding voxels on one side of a plane..."<< std::endl;
      thresholdWithRespectToPlane->Update();
    }
    catch (itk::ExceptionObject &ex)
    { 
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
    }

    pipeITKImageDataConnector = thresholdWithRespectToPlane->GetOutput();
  }
  else
    pipeITKImageDataConnector = setBoundary->GetOutput();


  // Create the ITK to VTK filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ImageToVTKImageFilter< InputImageType > ImageToVTKImageFilterType;

  ImageToVTKImageFilterType::Pointer convertITKtoVTK = ImageToVTKImageFilterType::New();

  convertITKtoVTK->SetInput( pipeITKImageDataConnector );

  try
  { 
    std::cout << "Converting the image to VTK format..." << std::endl;
    convertITKtoVTK->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }

  pipeVTKImageDataConnector = convertITKtoVTK->GetOutput();


  // Apply the Marching Cubes algorithm
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  vtkSmartPointer<vtkMarchingCubes> surfaceExtractor = vtkMarchingCubes::New();

  if ( distTransThresh )
    // The threshold 'd' corresponds to voxels which are sqrt(d) voxels
    // from the region grown boundary (negative = external, positive =
    // internal)
    surfaceExtractor->SetValue(0, distTransThresh);
  else
    // This is half the region groeing value
    surfaceExtractor->SetValue(0, 500);

  surfaceExtractor->SetInput((vtkDataObject *) pipeVTKImageDataConnector);
  pipeVTKPolyDataConnector = surfaceExtractor->GetOutput();


  if (verbose) {
    surfaceExtractor->Update();

    std::cout << std::endl << "Extracted surface data:" << std::endl;
    polyDataInfo(pipeVTKPolyDataConnector);
  }


  // Create triangles from the (assumed) polygonal data
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkTriangleFilter::New();
  triangleFilter->SetInput(pipeVTKPolyDataConnector);
  pipeVTKPolyDataConnector = triangleFilter->GetOutput();

  if (verbose) {
    triangleFilter->Update();

    std::cout << std::endl << "Polygonal data converted to triangles:" << std::endl;
    polyDataInfo(pipeVTKPolyDataConnector);
  }


  // Pre-decimation smoothing
  // ~~~~~~~~~~~~~~~~~~~~~~~~

  if (preSmooth) {
    vtkSmartPointer<vtkWindowedSincPolyDataFilter> preSmoothingFilter = vtkWindowedSincPolyDataFilter::New();
 
    preSmoothingFilter->BoundarySmoothingOff();

    if (featureAngle) {
      preSmoothingFilter->SetFeatureAngle(featureAngle);
      preSmoothingFilter->FeatureEdgeSmoothingOn();
    }
 
    preSmoothingFilter->SetNumberOfIterations(niterations);
    preSmoothingFilter->SetPassBand(bandwidth);
 
    preSmoothingFilter->SetInput(pipeVTKPolyDataConnector);
    pipeVTKPolyDataConnector = preSmoothingFilter->GetOutput();
  }


  // Decimate the polygonal data
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (decimation) {

    if ( flgDecimatePro ) {
      
      vtkSmartPointer<vtkDecimatePro> decimator = vtkDecimatePro::New();
      
      decimator->SetTargetReduction( decimation );
      decimator->PreserveTopologyOn();
      decimator->SplittingOff();
      decimator->BoundaryVertexDeletionOff();
      
      if (featureAngle) decimator->SetFeatureAngle( featureAngle );
      
      decimator->SetInput( pipeVTKPolyDataConnector );

      pipeVTKPolyDataConnector = decimator->GetOutput();
      
      if (verbose) {
	decimator->Update();
	
	std::cout << std::endl << "Decimated triangles:" << std::endl;
	polyDataInfo(pipeVTKPolyDataConnector);
      }
    }

    else {

      vtkSmartPointer<vtkQuadricDecimation> decimatorQD = vtkQuadricDecimation::New();
      
      decimatorQD->SetTargetReduction( decimation );
      decimatorQD->SetInput( pipeVTKPolyDataConnector );
      
      pipeVTKPolyDataConnector = decimatorQD->GetOutput();

      if (verbose) {
	decimatorQD->Update();
	
	std::cout << std::endl << "Decimated triangles:" << std::endl;
	polyDataInfo(pipeVTKPolyDataConnector);
      }
    }
  }

  
  // Post-decimation smoothing
  // ~~~~~~~~~~~~~~~~~~~~~~~~

  if (postSmooth) {
    vtkSmartPointer<vtkWindowedSincPolyDataFilter> postSmoothingFilter = vtkWindowedSincPolyDataFilter::New();
 
    postSmoothingFilter->BoundarySmoothingOff();

    if (featureAngle) {
      postSmoothingFilter->SetFeatureAngle(featureAngle);
      postSmoothingFilter->FeatureEdgeSmoothingOn();
    }
 
    postSmoothingFilter->SetNumberOfIterations(niterations);
    postSmoothingFilter->SetPassBand(bandwidth);
    
    postSmoothingFilter->SetInput(pipeVTKPolyDataConnector);
    pipeVTKPolyDataConnector = postSmoothingFilter->GetOutput();
  }


  // Write the created vtk surface to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputPolydata.length() > 0 ) {

    vtkSmartPointer<vtkPolyDataWriter> writer3D = vtkPolyDataWriter::New();
    writer3D->SetFileName( fileOutputPolydata.c_str() );
    writer3D->SetInput(pipeVTKPolyDataConnector);

    if (flgTextOutput)
      writer3D->SetFileType(VTK_ASCII);
    else
      writer3D->SetFileType(VTK_BINARY);

    writer3D->Write();

    if (verbose) 
    {
      std::cout << "Polydata written to VTK file: " << fileOutputPolydata.c_str() << std::endl;
    }
  }

  if ( fileOutputSTL.length() > 0 ) {

    vtkSmartPointer<vtkSTLWriter> writer3D = vtkSTLWriter::New();
    writer3D->SetFileName( fileOutputSTL.c_str() );
    writer3D->SetInput(pipeVTKPolyDataConnector);

    if (flgTextOutput)
      writer3D->SetFileType(VTK_ASCII);
    else
      writer3D->SetFileType(VTK_BINARY);

    writer3D->Write();

    if (verbose) 
      std::cout << "Polydata written to STL file: " << fileOutputPolydata.c_str() << std::endl;
  }

  if ( fileOutputSMESH.length() > 0 ) {
    WritePolydataAsTetgenSmeshFile( fileOutputSMESH.c_str(), pipeVTKPolyDataConnector );

    if (verbose) 
      std::cout << "Surface data written to SMESH file: " << fileOutputSMESH.c_str() << std::endl;
  }


  // Create surface normals to allow Goraud shading
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  vtkSmartPointer<vtkPolyDataNormals> normals = vtkPolyDataNormals::New();

  normals->SplittingOff();

  normals->SetInput(pipeVTKPolyDataConnector);
  pipeVTKPolyDataConnector = normals->GetOutput();


  // Map 3D volume to graphics library
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkSmartPointer<vtkPolyDataMapper> map3D = vtkPolyDataMapper::New();

  map3D->ScalarVisibilityOff();

  map3D->SetInput(pipeVTKPolyDataConnector);


  // Create the renderer, the render window, and the interactor. The renderer
  // draws into the render window, the interactor enables mouse- and 
  // keyboard-based interaction with the scene.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkSmartPointer<vtkRenderer>     ren    = vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renWin = vtkSmartPointer<vtkRenderWindow>::New();
  renWin->AddRenderer(ren);


  // Create actor
  // ~~~~~~~~~~~~

  vtkSmartPointer<vtkActor> actor3D = vtkActor::New();

  actor3D->SetMapper(map3D);
  actor3D->GetProperty()->SetColor(1, 1, 1);
  actor3D->GetProperty()->SetRepresentationToWireframe();

  ren->AddActor(actor3D);
  


  // Render the object
  // ~~~~~~~~~~~~~~~~~
    
  renWin->Render(); 


  // Start interactor
  // ~~~~~~~~~~~~~~~~
    
  if (! noInteractor) {
    vtkSmartPointer<vtkRenderWindowInteractor> iren = vtkRenderWindowInteractor::New(); 
    iren->SetRenderWindow(renWin); 

    vtkSmartPointer<vtkInteractorStyleTrackballCamera> istyle = vtkInteractorStyleTrackballCamera::New();
    iren->SetInteractorStyle(istyle);

    iren->Initialize(); 
    iren->Start(); 
  }

  return EXIT_SUCCESS;
}


// ----------------------------------------------------------
// polyDataInfo(vtkPolyData *polyData) 
// ----------------------------------------------------------


void polyDataInfo(vtkPolyData *polyData) 
{
  if (polyData) {
    std::cout << "   Number of vertices: " 
	 << polyData->GetNumberOfVerts() << std::endl;

    std::cout << "   Number of lines:    " 
	 << polyData->GetNumberOfLines() << std::endl;
    
    std::cout << "   Number of cells:    " 
	 << polyData->GetNumberOfCells() << std::endl;
    
    std::cout << "   Number of polygons: " 
	 << polyData->GetNumberOfPolys() << std::endl;
    
    std::cout << "   Number of strips:   " 
	 << polyData->GetNumberOfStrips() << std::endl;
  }
}


// ----------------------------------------------------------
// Write the poly data as a tetgen 'smesh' file
// ----------------------------------------------------------

void WritePolydataAsTetgenSmeshFile( const char *smesh_file, vtkPolyData *surface )
{

//  vtkCellArray *vertices = surface->GetVerts();

  ostream *fp = new ofstream(smesh_file, ios::out);
  

  if (fp->fail()) {
    std::cerr << "Could not open smesh file: " << smesh_file;
    return;
  }
  
  vtkPoints *points = surface->GetPoints();
  int numPts = points->GetNumberOfPoints();

  *fp << "# Node count, 3 dim, no attribute, no boundary marker" << endl;
  *fp << numPts << " " << 3 << " " << 0 << " " << 0 << endl;
  *fp << "# Node index, node coordinates" << endl;


  // Write the point data

  vtkDataArray *dataArray = points->GetData();

  switch (points->GetDataType()) {
  case VTK_FLOAT: 
  {
    float *data = ((vtkFloatArray *) dataArray)->GetPointer(0);
    int i, j;
    char str[1024];
      
    for (j=0; j<numPts; j++) {

      *fp << j << "\t";

      for (i=0; i<3; i++) {

	sprintf (str, "%g", *data++); 
	*fp << str; 
	if (i < 2) *fp << " "; 
      }

      *fp << endl;
    }
      
  }
  break;


  default:
  {
    std::cerr << "Could not write smesh data type";
    return;
  }

  }


  // Write the polygons

  vtkCellArray *cells = surface->GetPolys();

  int ncells=cells->GetNumberOfCells();
//  int size=cells->GetNumberOfConnectivityEntries();

  *fp << "# Facet listings" << endl;
  *fp << "# Number of facets, no boundary marker" << endl;
  *fp << ncells << " " << 0 << endl;
  *fp << "# Number of nodes defining the facet, node indices" << endl;

  if (ncells < 1) {
    std::cerr << "No cells in surface mesh, SMESH file save aborted." ;
    return;
  }

  int j;
  vtkIdType *pts = 0;
  vtkIdType npts = 0;
  
  for (cells->InitTraversal(); cells->GetNextCell(npts, pts); ) {

    *fp << (int) npts << "\t";

    for (j=0; j<npts; j++)
      *fp << (int) pts[j] << "\t";
    
    *fp << "\n";
  }
  
  *fp << "# No holes" << endl;
  *fp << 0 << endl;
  *fp << "# No region attributes" << endl;
  *fp << 0 << endl;

  delete fp;
}
