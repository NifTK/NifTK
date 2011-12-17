/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: ad  $
 $Date:: $
 $Rev:: $

 Copyright (c) UCL : See the file NifTKCopyright.txt in the top level 
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <vector>
#include <fstream>

#include "ConversionUtils.h"
#include "itkTransformFileReader.h"
#include "itkImageFileReader.h"
#include "itkImageRegistrationFactory.h"
#include "itkEulerAffineTransform.h"
#include "itkImageRegionConstIteratorWithIndex.h"



using namespace std;

void Usage(char *exec)
{
    cout << "  " << endl
	      << "  Compute the target registration error for a transformation" << endl
	      << "  given a set of landmarks in the target/fixed and source/moving images." << endl << endl
	      << "  NifTK, Copyright 2009" << endl << endl

	      << "*** [Error using landmarks] ***" << endl << endl
	      << "  " << exec << " -reg Transform -tp Points -sp Points [options]" << endl << "  " << endl
	      << "    -reg <filename>       The affine registration transformation" << endl
	      << "    -tp <filename>        Landmarks coordinates in the target/fixed image" << endl  
	      << "    -sp <filename>        Landmarks coordinates in the source/moving image" << endl  

	      << "*** [Comparison to ground truth transformation] ***" << endl << endl
	      << "  " << exec << " -reg Transform -gt Transform [options]" << endl << "  " << endl
	      << "    -reg <filename>       The affine registration transformation" << endl
	      << "    -gt <filename>        The affine ground truth transformation" << endl  
	      << "    -im <filename>        Image mask which has non-zero voxels for error calc." << endl  

	      << "*** [options]   ***" << endl << endl
	      << "    -invert               Invert the registration affine transformation" << endl  
	      << endl << endl;
}

struct arguments
{
  bool flgDebug;		         // Debugging output
  bool flgInvertAffineTransformation;    // Invert the affine and non-linear transformation? 

  string fileAffineRegnTransformation;   // The affine registration transformation file 
  string fileGroundTruthTransformation;  // The affine ground truth transformation file 

  string fileTargetLandmarks; // The target image landmark coordinates file
  string fileSourceLandmarks; // The source image landmark coordinates file

  string fileImageMask;	 // The image mask which has non-zero voxels for error calc.
};


template <class LandmarkType> 
void ReadLandmarks(string &fileLandmarks, vector<LandmarkType> &targetLandmarks);



// Templated main() routine to enable factory instantiation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <int Dimension, class PixelType> 
int DoMain(char *exec, arguments args)
{ 
  typedef typename itk::Image< PixelType, Dimension >  InputImageType; 
  typedef typename itk::EulerAffineTransform<double, Dimension, Dimension> EulerAffineTransformType;

  typedef typename EulerAffineTransformType::InputPointType LandmarkType;
  typedef typename EulerAffineTransformType::OutputPointType TransformedLandmarkType;

  typedef itk::ImageFileReader< InputImageType >  InputImageReaderType;
  typename InputImageReaderType::Pointer inputImageReader;

  double tre = 0.;
  double misreg = 0.;

  vector<LandmarkType> targetLandmarks;
  vector<LandmarkType> sourceLandmarks;

  EulerAffineTransformType *regnTransform = 0;
  EulerAffineTransformType *groundTruthTransform = 0;

 
  // Create the factory

  typedef typename itk::ImageRegistrationFactory<InputImageType, Dimension, double> FactoryType;
  
  typename FactoryType::Pointer factory = FactoryType::New();

  typename FactoryType::TransformType::Pointer affineTransform; 
  
  
  // Read the image ROI mask

  if (args.fileImageMask.length() != 0) {

    inputImageReader  = InputImageReaderType::New();

    inputImageReader->SetFileName( args.fileImageMask );

    try { 
      std::cout << "Reading input 3D volume: " << args.fileImageMask;
      inputImageReader->Update();
      std::cout << "Done";
    } 
    catch( itk::ExceptionObject & err ) { 
      std::cerr << "ERROR: Failed to load input image: " << err << std::endl; 
      return EXIT_FAILURE;
    }                
  }


  // Read the registration transformation

  if (args.fileAffineRegnTransformation.length() != 0) {

    try {
      
      cout << "Creating global transform from:" << args.fileAffineRegnTransformation << endl; 
      affineTransform = factory->CreateTransform(args.fileAffineRegnTransformation);
      cout << "Done" << endl; 
    }  
    
    catch (itk::ExceptionObject& exceptionObject) {
      
      cerr << "Failed to load global tranform:" << exceptionObject << endl;
      return EXIT_FAILURE; 
    }
    
    regnTransform = dynamic_cast<itk::EulerAffineTransform<double, Dimension, Dimension>*>(affineTransform.GetPointer()); 
    
    cout << regnTransform->GetFullAffineMatrix() << endl; 

    if (args.flgInvertAffineTransformation) {

      regnTransform->InvertTransformationMatrix(); 
      cout << "inverted:" << endl << regnTransform->GetFullAffineMatrix() << endl; 
    }
  }

  // Read the ground truth transformation

  if (args.fileGroundTruthTransformation.length() != 0) {

    try {
      
      cout << "Creating ground truth transform from:" << args.fileGroundTruthTransformation << endl; 
      affineTransform = factory->CreateTransform(args.fileGroundTruthTransformation);
      cout << "Done" << endl; 
    }  
    
    catch (itk::ExceptionObject& exceptionObject) {
      
      cerr << "Failed to load global tranform:" << exceptionObject << endl;
      return EXIT_FAILURE; 
    }
    
    groundTruthTransform = dynamic_cast<itk::EulerAffineTransform<double, Dimension, Dimension>*>(affineTransform.GetPointer()); 
    
    cout << groundTruthTransform->GetFullAffineMatrix() << endl; 
  }

  // Read the landmarks

  if (args.fileTargetLandmarks.length() != 0) 
    ReadLandmarks<LandmarkType>(args.fileTargetLandmarks, targetLandmarks);

  if (args.fileSourceLandmarks.length() != 0) 
    ReadLandmarks<LandmarkType>(args.fileSourceLandmarks, sourceLandmarks);


  // Iterate over the landmarks computing the target registration error
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ((targetLandmarks.size() > 0) && (sourceLandmarks.size() > 0) && regnTransform
      && ! ((args.fileImageMask.length() != 0) || groundTruthTransform)) {

    typename LandmarkType::VectorType vecDistance;
    TransformedLandmarkType transformedPoint;

    unsigned int i;

    if (targetLandmarks.size() != sourceLandmarks.size()) {
      std::cerr <<"Number of source and target landmarks differ.";
      return EXIT_FAILURE;   
    }

    // Calculate the initial misregistration

    for (i=0; i<targetLandmarks.size(); i++) {

      vecDistance = targetLandmarks[i] - sourceLandmarks[i];

      if (args.flgDebug)
	cout << "Initial: " << i << " " << targetLandmarks[i] << " - " << sourceLandmarks[i] 
	     <<  " = " << vecDistance <<  " = " << vecDistance.GetNorm() << endl;

      misreg += vecDistance.GetNorm();
    }
    misreg /= (double) targetLandmarks.size();

    // Calculate the tre

    for (i=0; i<targetLandmarks.size(); i++) {
      
      regnTransform->TransformPoint(targetLandmarks[i], transformedPoint);

      vecDistance = transformedPoint - sourceLandmarks[i];

      if (args.flgDebug)
	cout << "Registration: " << i << " " << transformedPoint << " - " << sourceLandmarks[i] 
	     <<  " = " << vecDistance <<  " = " << vecDistance.GetNorm() << endl;

      tre += vecDistance.GetNorm();
    
  
    }
    tre /= (double) targetLandmarks.size();

    cout << "Misregistration error = " << misreg << " mm" << endl
	 << "Target registration error = " << tre << " mm" << endl;

  }


  // Calculate registration error given a ground truth transformation
  // and an image mask
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  else if (((args.fileImageMask.length() != 0) && regnTransform && groundTruthTransform)
	   && ! ((targetLandmarks.size() > 0) || (sourceLandmarks.size() > 0))) {

    typename InputImageType::IndexType index;
    typename InputImageType::PointType point, regPoint, gndTruthPoint;
    typename InputImageType::PointType::VectorType vecDistance;

    typedef itk::ImageRegionConstIteratorWithIndex< InputImageType > ConstIteratorType;
    ConstIteratorType imIterator(inputImageReader->GetOutput(), inputImageReader->GetOutput()->GetLargestPossibleRegion());

    int i = 0;
    for (imIterator.GoToBegin(); ! imIterator.IsAtEnd(); ++imIterator) {

      if ( imIterator.Get() != 0) {

	index = imIterator.GetIndex();

	inputImageReader->GetOutput()->TransformIndexToPhysicalPoint(index, point);

	regnTransform->TransformPoint(point, regPoint);
	groundTruthTransform->TransformPoint(point, gndTruthPoint);
	
	vecDistance = gndTruthPoint - regPoint;
	tre += vecDistance.GetNorm();
	
	vecDistance = gndTruthPoint - point;
	misreg += vecDistance.GetNorm();

	if (args.flgDebug)
	  cout << i << " " << index << " pt: " << point << " gt: " << gndTruthPoint << " reg: " << regPoint << endl;

	i++;
      }
    }

    tre /= (double) i;
    misreg /= (double) i;

    cout << "Misregistration error = " << misreg << " mm" << endl
	 << "Target registration error = " << tre << " mm" << endl;      
  }


  // Incorrect command line args
  
  else {
    
    Usage(exec);
    return EXIT_FAILURE;
  }

  
  std::cout << "Done";
  
  return EXIT_SUCCESS;   
}




// main() routine to read command line arguments
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// \brief Create an affine transformation with various formats.


int main(int argc, char** argv)
{

  // To pass around command line args
  struct arguments args;

  // Parse command line args

  args.flgInvertAffineTransformation = false;
  args.flgDebug = false;

  for(int i=1; i < argc; i++){

    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 
       || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }

    else if(strcmp(argv[i], "-v") == 0) {
      cout << "Verbose output enabled" << endl;
    }

    else if(strcmp(argv[i], "-dbg") == 0) {
      args.flgDebug = true;
      cout << "Debugging output enabled" << endl;
    }

    else if(strcmp(argv[i], "-invert") == 0) {
      args.flgInvertAffineTransformation = true;
      std::cout << "Set -invert= true";
    }

    else if(strcmp(argv[i], "-reg") == 0) {
      args.fileAffineRegnTransformation = argv[++i];
      std::cout << "Set -reg=" << args.fileAffineRegnTransformation;
    }

    else if(strcmp(argv[i], "-gt") == 0) {
      args.fileGroundTruthTransformation = argv[++i];
      std::cout << "Set -gt=" << args.fileGroundTruthTransformation;
    }

    else if(strcmp(argv[i], "-im") == 0) {
      args.fileImageMask = argv[++i];
      std::cout << "Set -im=" << args.fileImageMask;
    }

    else if(strcmp(argv[i], "-tp") == 0) {
      args.fileTargetLandmarks = argv[++i];
      std::cout << "Set -tp=" << args.fileTargetLandmarks;
    }
    else if(strcmp(argv[i], "-sp") == 0) {
      args.fileSourceLandmarks = argv[++i];
      std::cout << "Set -sp=" << args.fileSourceLandmarks;
    }

    else {
      cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << endl;
      return -1;
    }            
  }

  cout << endl;

  // Call the templated routine to do the processing

  return DoMain<3, float>(argv[0], args);
}


// Read a set of landmarks from a file
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <class LandmarkType> 
void ReadLandmarks(string &fileLandmarks, vector<LandmarkType> &landmarks) 
{
  unsigned int i;
  ifstream fin;
  LandmarkType landmark;


  fin.open(fileLandmarks.c_str());

  if ((! fin) || fin.bad()) {
    std::cerr <<"Could not open landmarks file.";
    return;
  }

  while (fin.eof() == 0) {

    fin >> landmark[0];
    fin >> landmark[1];
    fin >> landmark[2];

    if (fin.eof()) break;

    cout << ++i << " "
	 << landmark[0] << " " 
	 << landmark[1] << " " 
	 << landmark[2] << endl; 

    landmarks.push_back(landmark);
  }


  fin.close();
}

