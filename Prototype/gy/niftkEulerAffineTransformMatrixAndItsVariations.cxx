/*=============================================================================

NifTK: An image processing toolkit jointly developed by the
Dementia Research Centre, and the Centre For Medical Image Computing
at University College London.

See:        http://dementia.ion.ucl.ac.uk/
http://cmic.cs.ucl.ac.uk/
http://www.ucl.ac.uk/

Last Changed      : $Date: 2010-06-04 15:11:19 +0100 (Fri, 04 Jun 2010) $
Revision          : $Revision: 3349 $
Last modified by  : $Author: jhh, gy $

Original author   : j.hipwell@ucl.ac.uk

Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notices for more information.

============================================================================*/

#include "itkLogHelper.h"
#include "ConversionUtils.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkEulerAffineTransform.h"
#include "itkTransformFactory.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"
#include "itkEulerAffineTransformMatrixAndItsVariations.h"


void Usage(char *exec)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl
    << "  Compute an affine transformation of a 3D image volume using affine transformation matrix." << std::endl << std::endl

    << "  " << exec 
    << " -im InputImage -s3D Input3DimageSize -o OutputAffineTransformedImage"
    << std::endl << "  " << std::endl

    << "*** [mandatory] ***" << std::endl << std::endl
    << "    -im   <filename>                  Input 3D image volume " << std::endl
    << "    -s3D  <int> <int> <int>        	  Input 3D image volume size [100 x 100 x100] " << std::endl
    << "    -o   <filename>                   Output 3D affine transformed image" << std::endl

    << "*** [options]   ***" << std::endl << std::endl
    << "    -v                                Output verbose info" << std::endl
    << "    -dbg                              Output debugging info" << std::endl
    << "    -ip  <doublex12>                  Input the 12 transformation parameters directly" << std::endl << std::endl;
}


/**
 * \brief Project a 3D image volume into 3D.
 */
int main(int argc, char** argv)
{
  typedef double 																			TScalarType;
  typedef float 																			IntensityType;
  typedef vnl_sparse_matrix<TScalarType>           		SparseMatrixType;
  typedef vnl_vector<TScalarType>                    	VectorType;

  typedef itk::EulerAffineTransformMatrixAndItsVariations< TScalarType > 													AffineTransformerType;
  typedef AffineTransformerType::EulerAffineTransformType 																				EulerAffineTransformType;
  EulerAffineTransformType::ParametersType 																												parameters;

  bool flgDebug 	= false;

  std::string fileInputImage3D;
  std::string fileOutputImage3D;
  typedef itk::Image<IntensityType, 3>            																								InputImageType;
  typedef InputImageType::Pointer      																														InputImagePointer;
  typedef InputImageType::ConstPointer 																														InputImageConstPointer;
  typedef InputImageType::RegionType   																														InputImageRegionType;
  typedef InputImageType::PixelType    																														InputImagePixelType;
  typedef InputImageType::SizeType     																														InputImageSizeType;
  typedef InputImageType::SpacingType  																														InputImageSpacingType;
  typedef InputImageType::PointType   																														InputImagePointType;
  typedef InputImageType::IndexType   																														InputImageIndexType;
  typedef itk::ImageFileReader< InputImageType >  																								InputImageReaderType;

  // Initialise the affine parameters
  parameters.SetSize(12); // parameters 0-2: translation; 3-5: rotation; 6-8 scaling; 9-11: skew.

  parameters.Fill(0.);

  parameters[0]  = 2.0;
  parameters[1]  = 0.0;
  parameters[2]  = 0.0;
  parameters[3]  = 0.0;
  parameters[4]  = 0.0;
  parameters[5]  = 0.0;
  parameters[6]  = 1.0;
  parameters[7]  = 1.0;
  parameters[8]  = 1.0;
  parameters[9]  = 0.0;
  parameters[10] = 0.0;
  parameters[11] = 0.0;


  std::cout << std::endl << argv[0] << std::endl << std::endl;

  // The dimensions in pixels of the 3D image
  AffineTransformerType::VolumeSizeType   nVoxels3D;
  // The resolution in mm of the 3D image
  InputImageSpacingType  									spacing3D;
  // The origin in mm of the 3D image
  InputImagePointType											origin3D;

  spacing3D[0] = 1.;
  spacing3D[1] = 1.;
  spacing3D[2] = 1.;

  origin3D[0] = 0.;
  origin3D[1] = 0.;
  origin3D[2] = 0.;


  // Parse command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~

  for(int i=1; i < argc; i++){

    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 
        || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }

    else if(strcmp(argv[i], "-v") == 0) {
      std::cout << "Verbose output enabled" << std::endl;
    }

    else if(strcmp(argv[i], "-dbg") == 0) {
      flgDebug = true;
     std:: cout << "Debugging output enabled" << std::endl;
    }

    else if(strcmp(argv[i], "-s3D") == 0) {
      nVoxels3D[0] = atoi(argv[++i]);
      nVoxels3D[1] = atoi(argv[++i]);
      nVoxels3D[2] = atoi(argv[++i]);
      std::cout << std::string("Set -s3D=")
          << niftk::ConvertToString((int) nVoxels3D[0]) << " "
          << niftk::ConvertToString((int) nVoxels3D[1]) << " "
          << niftk::ConvertToString((int) nVoxels3D[2]);
    }

    else if(strcmp(argv[i], "-im") == 0) {
      fileInputImage3D = argv[++i];
      std::cout << std::string("Set -im=") << fileInputImage3D;
    }

    else if(strcmp(argv[i], "-o") == 0) {
      fileOutputImage3D = argv[++i];
      std::cout << std::string("Set -o=") << fileOutputImage3D;
    }

    else if(strcmp(argv[i], "-ip") == 0) {
      parameters[0] 	= atof(argv[++i]);
      parameters[1] 	= atof(argv[++i]);
      parameters[2] 	= atof(argv[++i]);
      parameters[3] 	= atof(argv[++i]);
      parameters[4] 	= atof(argv[++i]);
      parameters[5] 	= atof(argv[++i]);
      parameters[6] 	= atof(argv[++i]);
      parameters[7] 	= atof(argv[++i]);
      parameters[8] 	= atof(argv[++i]);
      parameters[9] 	= atof(argv[++i]);
      parameters[10] 	= atof(argv[++i]);
      parameters[11] 	= atof(argv[++i]);
      std::cout << std::string("Set -ip=")
          << niftk::ConvertToString( parameters[0])  << " "
          << niftk::ConvertToString( parameters[1])  << " "
          << niftk::ConvertToString( parameters[2])  << " "
          << niftk::ConvertToString( parameters[3])  << " "
          << niftk::ConvertToString( parameters[4])  << " "
          << niftk::ConvertToString( parameters[5])  << " "
          << niftk::ConvertToString( parameters[6])  << " "
          << niftk::ConvertToString( parameters[7])  << " "
          << niftk::ConvertToString( parameters[8])  << " "
          << niftk::ConvertToString( parameters[9])  << " "
          << niftk::ConvertToString( parameters[10]) << " "
          << niftk::ConvertToString( parameters[11]);
    }

    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( 		 (nVoxels3D[0] 																	== 0) 
				|| (nVoxels3D[1] 																	== 0) 
				|| (nVoxels3D[2] 																	== 0) 
				|| (fileInputImage3D.length() 										== 0) 
				|| (fileOutputImage3D.length() 								    == 0) ) {
    Usage(argv[0]);
		std::cout << std::endl << "  -help for more options" << std::endl << std::endl;
    return EXIT_FAILURE;
  }             

  // Create the affine transformer
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  AffineTransformerType::Pointer affineTransformer = AffineTransformerType::New();

  unsigned long int totalSize = nVoxels3D[0]*nVoxels3D[1]*nVoxels3D[2];
  static SparseMatrixType affineMatrix(totalSize, totalSize);
  static SparseMatrixType affineMatrixTranspose(totalSize, totalSize);

  // Obtain the affine transformation matrix and its transpose
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  affineTransformer->GetAffineTransformationSparseMatrix(affineMatrix, nVoxels3D, parameters);
  affineTransformer->GetAffineTransformationSparseMatrixT(affineMatrix, affineMatrixTranspose, nVoxels3D);


#if 0
  // Print out the all the entries of the sparse affine matrix
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  affineMatrix.reset();
  std::ofstream sparseAffineFullMatrixFile("sparseAffineFullMatrix.txt");

 	for ( unsigned long int totalEntryIterRow = 0; totalEntryIterRow < affineMatrix.rows(); totalEntryIterRow++ )
		for ( unsigned long int totalEntryIterCol = 0; totalEntryIterCol < affineMatrix.cols(); totalEntryIterCol++ )
  		sparseAffineFullMatrixFile << affineMatrix(totalEntryIterRow, totalEntryIterCol) << " ";


  // Print out the all the entries of the transpose of the sparse affine matrix
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  affineMatrixTranspose.reset();
  std::ofstream sparseAffineTransposeFullMatrixFile("sparseAffineTransposeFullMatrix.txt");

 	for ( unsigned long int totalEntryIterRow = 0; totalEntryIterRow < affineMatrixTranspose.rows(); totalEntryIterRow++ )
		for ( unsigned long int totalEntryIterCol = 0; totalEntryIterCol < affineMatrixTranspose.cols(); totalEntryIterCol++ )
  		sparseAffineTransposeFullMatrixFile << affineMatrixTranspose(totalEntryIterRow, totalEntryIterCol) << " ";


  // Print out the non-zero entries
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  affineMatrix.reset();
  std::ofstream sparseAffineMatrixFile("sparseAffineMatrix.txt");
  sparseAffineMatrixFile << std::endl << "The non-zero entries of the affine matrix are: " << std::endl;

  unsigned long int rowIndex = 0;
  unsigned long int colIndex = 0;

  while ( affineMatrix.next() )
  {
  	rowIndex = affineMatrix.getrow();
  	colIndex = affineMatrix.getcolumn();

  if ( (rowIndex < affineMatrix.rows()) && (colIndex < affineMatrix.cols()) )	
  	sparseAffineMatrixFile << std::endl << "Row " << rowIndex << " and column " << colIndex << " is: " << affineMatrix.value() << std::endl;
  }

  // Print out the non-zero entries of the transpose
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  affineMatrixTranspose.reset();
  std::ofstream sparseAffineMatrixTFile("sparseAffineMatrixT.txt");
  sparseAffineMatrixTFile << std::endl << "The non-zero entries of the affine matrix transpose are: " << std::endl;

  unsigned long int rowIndexT = 0;
  unsigned long int colIndexT = 0;

  while ( affineMatrixTranspose.next() )
  {
  	rowIndexT = affineMatrixTranspose.getrow();
  	colIndexT = affineMatrixTranspose.getcolumn();

  if ( (rowIndexT < affineMatrixTranspose.rows()) && (colIndexT < affineMatrixTranspose.cols()) )	
  	sparseAffineMatrixTFile << std::endl << "Row " << rowIndexT << " and column " << colIndexT << " is: " << affineMatrixTranspose.value() << std::endl;
  } 
#endif


  // Load the input image volume
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileInputImage3D.length() != 0 )
  {

    InputImageReaderType::Pointer inputImageReader  = InputImageReaderType::New();

    inputImageReader->SetFileName( fileInputImage3D );

    try { 
      std::cout << std::string("Reading input 3D volume: ") <<  fileInputImage3D;
      inputImageReader->Update();
      std::cout << std::string("Done");
    } 
    catch( itk::ExceptionObject & err ) { 
      std::cerr << "ERROR: Failed to load input image: " << err << std::endl; 
      return EXIT_FAILURE;
    }

    InputImageConstPointer inImage = inputImageReader->GetOutput();

    // Covert the input image into the vnl vector form
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> ConstIteratorType;
    ConstIteratorType inputIterator( inImage, inImage->GetLargestPossibleRegion() );	

    VectorType inputImageVector(totalSize);

    unsigned long int voxel3D = 0;
    InputImagePixelType voxelValue;
    for ( inputIterator.GoToBegin(); !inputIterator.IsAtEnd(); ++inputIterator)
    {
      voxelValue = inputIterator.Get();
      inputImageVector.put(voxel3D, (double) voxelValue);

      voxel3D++;	 
    }

    std::ofstream inputImageVectorFile("inputImageVector.txt");
    inputImageVectorFile << inputImageVector << " ";

    // Calculate the matrix/vector multiplication in order to get the affine transformation
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert (!inputImageVector.is_zero());
    VectorType outputImageVector(totalSize);
    outputImageVector.fill(0.);

    // affineMatrix.mult(inputImageVector, outputImageVector);
    affineTransformer->CalculteMatrixVectorMultiplication(affineMatrix, inputImageVector, outputImageVector);

    std::ofstream vectorFile("vectorFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    vectorFile << outputImageVector << " ";

#if 0
    // Calculate the matrix/vector multiplication in order to get the ajoint affine transformation
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert (!outputImageVector.is_zero());
    VectorType outputImageVectorTwo(totalSize);
    outputImageVectorTwo.fill(0.);

    // affineMatrix.mult(inputImageVector, outputImageVector);
    affineTransformer->CalculteMatrixVectorMultiplication(affineMatrixTranspose, outputImageVector, outputImageVectorTwo);

    std::ofstream vectorTwoFile("vectorTwoFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    vectorTwoFile << outputImageVectorTwo << " ";

    // Calculate the matrix/vector multiplication in order to get the gradient vector
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert (!inputImageVector.is_zero());
    VectorType outputVectorGrad(totalSize);
    outputVectorGrad.fill(0.);

    affineTransformer->CalculteMatrixVectorGradient(nVoxels3D, inputImageVector, outputVectorGrad, parameters, 0);

    std::ofstream outputVectorGradFile("outputVectorGradFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    outputVectorGradFile << outputVectorGrad << " ";
#endif

  	// Type definitions for the output images
  	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  	typedef float 																																			OutputReconstructionType;

  	typedef itk::Image< OutputReconstructionType, 3 > 																	OutputImageType;
		typedef OutputImageType::Pointer 																										OutputImagePointer;
  	typedef OutputImageType::RegionType   																							OutputImageRegionType;
  	typedef OutputImageType::PixelType    																							OutputImagePixelType;
  	typedef OutputImageType::SizeType     																							OutputImageSizeType;
  	typedef OutputImageType::SpacingType  																							OutputImageSpacingType;
  	typedef OutputImageType::PointType   																								OutputImagePointType;
  	typedef OutputImageType::IndexType   																								OutputImageIndexType;
  	typedef itk::ImageRegionConstIteratorWithIndex<OutputImageType> 										OutputImageConstIteratorType;
  	typedef itk::ImageFileWriter< OutputImageType > 																		OutputImageWriterType;
  	typedef itk::Size<3> 					             																					OutputProjectionVolumeSizeType;

  	// Write the transformed image out
  	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  	if (fileOutputImage3D.length() > 0) {

			// Construct the output moving image and copy in the voxel values from the image vector
  		OutputImagePointer outputMovingImage = OutputImageType::New();

  		OutputImageIndexType outputStart;
  		outputStart[0] = 0; // first index on X
  		outputStart[1] = 0; // first index on Y
  		outputStart[2] = 0; // first index on Z

  		OutputImageRegionType 	outputRegion;
  		outputRegion.SetSize( 	nVoxels3D );
  		outputRegion.SetIndex( 	outputStart );

  		outputMovingImage->SetRegions( 	outputRegion );
  		outputMovingImage->SetOrigin( 	origin3D );
  		outputMovingImage->SetSpacing( 	spacing3D );

  		outputMovingImage->Allocate();

			unsigned int iVoxel = 0; 
  		OutputImageConstIteratorType outputMovingImageIter( outputMovingImage, outputMovingImage->GetLargestPossibleRegion() );
			for ( outputMovingImageIter.GoToBegin(); !outputMovingImageIter.IsAtEnd(); ++outputMovingImageIter )
			{			
				outputMovingImage->SetPixel( outputMovingImageIter.GetIndex(), ((float) outputImageVector.get(iVoxel)) );
				iVoxel++;
			}

  		// Then write the image

  		OutputImageWriterType::Pointer outputMovingImageWriter = OutputImageWriterType::New();

  		outputMovingImageWriter->SetFileName( fileOutputImage3D );
  		outputMovingImageWriter->SetInput( 		outputMovingImage );

  		try {
    		std::cout << std::string("Writing moving image output to file: ") << fileOutputImage3D;
    		outputMovingImageWriter->Update();
  		}
  		catch( itk::ExceptionObject & err ) {
    		std::cerr << "ERROR: Failed to write moving image output to file: " << fileOutputImage3D << "; " << err << std::endl;
    		return EXIT_FAILURE;
  		}
  
  	}

  }

  std::cout << std::string("Done");

  return EXIT_SUCCESS;   
}

