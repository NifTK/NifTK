/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <fstream>
#include <iomanip>

#include <itkLogHelper.h>

#include <niftkConversionUtils.h>
#include <niftkCommandLineParser.h>
#include <itkCommandLineHelper.h>

#include <itkImageFileWriter.h>
#include <itkThinPlateSplineScatteredDataPointSetToImageFilter.h>


/*!
 * \file niftkThinPlateSplineScatteredDataPointSetToImage.cxx
 * \page niftkThinPlateSplineScatteredDataPointSetToImage
 * \section niftkThinPlateSplineScatteredDataPointSetToImageSummary Computes a thin plate spline mask approximation to a set of landmarks.
 */

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output."},
  {OPT_SWITCH, "dbg", NULL, "Output debugging info."},

  {OPT_INTx3,   "size",   "nx,ny,nz", 
   "The size of the output volume [100 x 100 x 100]. For 2D data set nz = 1 or 0."},

  {OPT_FLOATx3, "res",    "rx,ry,rz", "The resolution of the output volume [1mm x 1mm x 1mm]"},
  {OPT_FLOATx3, "origin", "ox,oy,oz", "The origin of the output volume [0mm x 0mm x 0mm]"},

  {OPT_DOUBLE, "s", "stiffness", "The stiffness of the spline [ 1 ]"},

  {OPT_STRING|OPT_REQ, "o", "fileOut", "The file name of the output voxel planes."},
  {OPT_STRING|OPT_REQ, "i", "fileIn", "The file name of the input landmarks."},

  {OPT_DONE, NULL, NULL,
   "Program to compute a thin plate spline mask approximation to a set of landmarks.\n"
  }
};

enum {
  O_VERBOSE,
  O_DEBUG,

  O_SIZE,
  O_RES,
  O_ORIGIN,

  O_STIFFNESS,

  O_OUTPUT_FILE,
  O_INPUT_FILE
};


struct arguments
{
  bool flgVerbose;
  bool flgDebug;

  int size[3];

  float resolution[3];
  float origin[3];

  double stiffness;

  std::string fileOutput;
  std::string fileInput;

  arguments() {
    flgVerbose = false;
    flgDebug = false;

    size[0] = 100;
    size[1] = 100;
    size[2] = 100;

    resolution[0] = 1.;
    resolution[1] = 1.;
    resolution[2] = 1.;

    origin[0] = 0.;
    origin[1] = 0.;
    origin[2] = 0.;

    stiffness = 1.;
  }
};


template <int OutputDimension, class PixelType>
int DoMain(arguments args)
{
  typedef itk::Image< PixelType, OutputDimension >                 OutputImageType; 

  typedef itk::ImageFileWriter< OutputImageType >                  OutputImageWriterType;

  typename OutputImageType::RegionType  region;
  typename OutputImageType::IndexType   index;
  typename OutputImageType::SizeType    size;
  typename OutputImageType::SpacingType spacing;
  typename OutputImageType::PointType   origin;

  typedef itk::PointSet<double, OutputDimension> LandmarkPointSetType;

  typedef typename itk::ThinPlateSplineScatteredDataPointSetToImageFilter< LandmarkPointSetType, OutputImageType > ThinPlateSplineFilterType;


  // Read the input points
  // ~~~~~~~~~~~~~~~~~~~~~

  unsigned int iDim;
  unsigned int iLandmark;
  unsigned int nLandmarks;

  std::fstream fin;

  typedef typename ThinPlateSplineFilterType::LandmarkPointType LandmarkPointType;

  typename LandmarkPointSetType::Pointer PointSet = LandmarkPointSetType::New();

  LandmarkPointType point; 

  if ( args.flgVerbose ) {
    std::cout << "Opening landmarks file: " << args.fileInput << "..." << std::endl;
    std::cout.flush();
  }

  fin.open( args.fileInput.c_str(), std::ios::in );

  if ( fin.bad() || fin.fail() ) {
    std::cerr << "ERROR: Failed to open file: " << args.fileInput << std::endl;
    return EXIT_FAILURE;
  }

  iLandmark = 0;

  if ( args.flgVerbose )
  {
    std::cout << "Landmarks:" << std::endl;
  }

  while ( fin.eof() == 0 )
  {

    for ( iDim=0; iDim<OutputDimension; iDim++ )
    {
      fin >> point[iDim];
      if ( fin.eof() ) break;
    }

    if ( fin.eof() ) break;

    if ( args.flgVerbose )
    {
      std::cout << std::setw(8) << iLandmark << " "
                << point << std::endl;
    }

    PointSet->SetPoint( iLandmark, point );

    iLandmark++;
  }

  
  // Fit a thin plate spline to the points
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename ThinPlateSplineFilterType::Pointer filter = ThinPlateSplineFilterType::New();

  filter->SetInput( PointSet );

  for ( iDim=0; iDim<OutputDimension; iDim++ )
  {
    index[iDim] = 0;
    size[iDim] = args.size[ iDim ];
    spacing[iDim] = args.resolution[ iDim ];
    origin[iDim] = args.origin[ iDim ];
  }

  region.SetSize( size );
  region.SetIndex( index );

  filter->SetSpacing( spacing );
  filter->SetOrigin(  origin );
  filter->SetSize( size );

  filter->SetStiffness( args.stiffness );

  try
  {  
    std::cout << "Computing thin plate spline mask" << std::endl;
    filter->Update();
  }                                                                                
  catch (itk::ExceptionObject &e)                                                  
  {                                                                                
    std::cerr << e << std::endl;                                                   
    return EXIT_FAILURE;                                                           
  }                                                                                


  // Write the image out
  // ~~~~~~~~~~~~~~~~~~~

  if ( args.fileOutput.length() != 0 ) 
  {                                        
    typename OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

    writer->SetFileName( args.fileOutput );
    writer->SetInput( filter->GetOutput() );               

    try
    {  
      std::cout << "Writing the thin plate spline mask to image: " << args.fileOutput << std::endl;
      writer->Update();                                                              
    }                                                                                
    catch (itk::ExceptionObject &e)                                                  
    {                                                                                
      std::cerr << e << std::endl;                                                   
      return EXIT_FAILURE;                                                           
    }                                                                                
  }                                                                                  
   


  return EXIT_SUCCESS;
}

/**
 * \brief Determines the input image dimension and pixel type.
 */
int main(int argc, char** argv)
{
  int result;

  int *clo_size = 0;		// The size of the reconstructed volume

  float *clo_res = 0;		// The resolution of the reconstructed volume
  float *clo_origin = 0;		// The origin of the reconstructed volume

  struct arguments args;

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_VERBOSE, args.flgVerbose );
  CommandLineOptions.GetArgument( O_DEBUG, args.flgDebug );

  if (CommandLineOptions.GetArgument(O_SIZE, clo_size)) {
    args.size[0] = clo_size[0];
    args.size[1] = clo_size[1];
    args.size[2] = clo_size[2];
  }

  if (CommandLineOptions.GetArgument(O_RES, clo_res)) {
    args.resolution[0] = clo_res[0];
    args.resolution[1] = clo_res[1];
    args.resolution[2] = clo_res[2];
  }

  if (CommandLineOptions.GetArgument(O_ORIGIN, clo_origin)) {
    args.origin[0] = clo_origin[0];
    args.origin[1] = clo_origin[1];
    args.origin[2] = clo_origin[2];
  }

  CommandLineOptions.GetArgument( O_STIFFNESS, args.stiffness );

  CommandLineOptions.GetArgument( O_OUTPUT_FILE, args.fileOutput );
  CommandLineOptions.GetArgument( O_INPUT_FILE, args.fileInput );


  if ( args.size[2] > 1 )
  {

    result = DoMain<3, float>( args );  

  }
  else
  {

    result = DoMain<2, float>( args );  

  }

  return result;
}
