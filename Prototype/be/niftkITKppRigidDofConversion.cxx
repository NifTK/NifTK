/*
 * Author:  B. Eiben
 * Date:    4th Feb. 2011
 * Purpose: This software intends to translate the degrees of freedom files (currently rigid only)
 *          created by rreg (tik++ software package) into a homogenous matrix which could be used
 *          by the validation software.
 *
 */



/* standard includes */
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>

/* toolkit includes */
#include "CommandLineParser.h"
#include "ITKppDofFileReader.h"
#include "ITKppImage2WorldMatrixCalculator.h"

/* itk includes */
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkMatrix.h"
#include "itkAffineTransform.h"
#include "itkTransformFileWriter.h"
#include "itkImageRegistrationFactory.h"



/**
 * Command line parer sturcture.
 */
struct niftk::CommandLineArgumentDescription clArgList[] =
{
    { OPT_STRING| OPT_REQ, "dof",      "filename", "Filename which contains the degrees of freedom." },
    { OPT_STRING| OPT_REQ, "target",   "filename", "Filename of the target image." },
	{ OPT_STRING,          "mat",      "filename", "Output matrix." },//
	{ OPT_STRING,          "itk",      "filename", "Output itk transformation file (itk::AffineTransform)." },
	{ OPT_STRING,          "ucl",      "filename", "Output niftk transformation file (itk::AffineTransform with 16 parameters)." },
	{ OPT_SWITCH,          "rot",      0,          "Interpret the matrix for a rotated coordinate system M' = R M R. "},
	{ OPT_DONE, NULL, NULL,            "Transforms the rigid or affine 3D dof file of the itk++ package into an homogeneous matrix for further"
                                       "processing. "
                                       "NOTE: The matrix given by itk++ is considerd around the rotation centre."
                                       "      This program corrects this assumption." }
};



/**
 * Helper enumeration for command line parser.
 */
enum
{
    O_DOF = 0,
    O_TARGET,
    O_MATRIX,
	O_ITK_TRANSFORM,
	O_UCL_TRANSFORM,
	O_ROT_MATRIX,
};



/**
 * Start the transformation.
 */
int main(int argc, char ** argv)
{
    std::string strDofIn;
    std::string strTargetImage;
    std::string strOutMatrix;
	std::string strITKTransformFile;
	std::string strUCLTransformFile;
	bool        bRotateMatrix;

	
    /*
     * Parse the command line
     */
    niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

    CommandLineOptions.GetArgument( O_DOF,           strDofIn            );
    CommandLineOptions.GetArgument( O_TARGET,        strTargetImage      );
    CommandLineOptions.GetArgument( O_MATRIX,        strOutMatrix        );
	CommandLineOptions.GetArgument( O_ITK_TRANSFORM, strITKTransformFile );
	CommandLineOptions.GetArgument( O_UCL_TRANSFORM, strUCLTransformFile );
	CommandLineOptions.GetArgument( O_ROT_MATRIX,    bRotateMatrix       );

    /*
     * Read the dof-file.
     */
    const unsigned int Dimension = 3;
    typedef ITKppDofFileReader<Dimension> DOFReaderType3D;
    typedef DOFReaderType3D::HomogenousMatrixType HomogenousMatrixType;
    DOFReaderType3D dofReader( strDofIn );
    dofReader.PrintDofs();
    dofReader.PrintHomogenousMatrix();
    HomogenousMatrixType transformMat = dofReader.GetHomogenousMatrix();

    /*
     * Read the image file.
     */
    typedef short                                 PixelType;
    typedef itk::Image<PixelType, Dimension>      InputImageType;
    typedef InputImageType::Pointer               InputImagePointerType;
    typedef itk::ImageFileReader<InputImageType>  ImageReaderType;
    typedef ImageReaderType::Pointer              ImageReaderPointerType;

    ImageReaderPointerType reader = ImageReaderType::New();
    reader->SetFileName( strTargetImage );


    try
    {
		reader->Update();
	}
    catch (itk::ExceptionObject e)
    {
    	std::cerr << "An error occurred while trying to read the image file: "  << std::endl
			      << "  -> " << strTargetImage << std::endl;
	}

    InputImagePointerType image = reader->GetOutput();


    /*
     * Generate the matrices
     */
    typedef ITKppImage2WorldMatrixCalculator< InputImageType,
    		                                  Dimension >               MatrixGeneratorType;

    MatrixGeneratorType  im2wGenerator( image );
    HomogenousMatrixType im2w = im2wGenerator.GetImage2WolrdTranslationOnlyMatrix();
    HomogenousMatrixType w2im = im2wGenerator.GetImage2WolrdInverseTranslationOnlyMatrix();




    /*
     * Do the translation
     */
    HomogenousMatrixType resultMat = w2im * transformMat * im2w;
    std::cout << "resulting matrix:" << std::endl << resultMat << std::endl;

	/*
	 * Rotate the matrix in a different coordinate system
	 */
	if ( bRotateMatrix )
	{
		HomogenousMatrixType rotMat;
		rotMat.SetIdentity();
		rotMat[0][0] = -1;
		rotMat[1][1] = -1;

		std::cout << "Created rotation matrix: "       << std::endl << rotMat    << std::endl;
		std::cout << "Result matrix before rotation: " << std::endl << resultMat << std::endl;
		resultMat = rotMat * resultMat * rotMat;

		std::cout << "Result matrix after rotation: " << std::endl << resultMat << std::endl;
	}


    /*
     * Write the result
     */

	if (! strOutMatrix.empty() )
	{
		std::ofstream outFile;
	    outFile.open( strOutMatrix.c_str(), std::ios_base::trunc );

		for (unsigned int j = 0;  j < Dimension+1;  ++j)
		{
			for(unsigned int i = 0;  i < Dimension+1; ++i)
			{
				outFile << std::setw(18) << std::setprecision(10) << resultMat(j,i) << " ";
			}
			outFile << std::endl;
		}
		outFile.close();
	}
	

	// Write the result to an itk transformation
	std::cout << "itk part: " << std::endl;

	typedef itk::AffineTransform< double, Dimension >   itkTransformType;
	typedef itkTransformType::Pointer                   itkTransformPointerType;
	typedef itkTransformType::ParametersType			itkParametersType;
	
	typedef itk::TransformFileWriter                    itkTransformWriterType;
	typedef itkTransformWriterType::Pointer             itkTransformWriterPointerType;
	
	itkTransformPointerType itkTransform = itkTransformType::New();
	itkTransform->SetIdentity();

	itkParametersType parameters = itkTransform->GetParameters();

	// first index of the matrix gives the row!
	parameters[ 0] = resultMat[0][0];
	parameters[ 1] = resultMat[0][1];
	parameters[ 2] = resultMat[0][2];
	parameters[ 3] = resultMat[1][0];
	parameters[ 4] = resultMat[1][1];
	parameters[ 5] = resultMat[1][2];
	parameters[ 6] = resultMat[2][0];
	parameters[ 7] = resultMat[2][1];
	parameters[ 8] = resultMat[2][2];
	parameters[ 9] = resultMat[0][3];
	parameters[10] = resultMat[1][3];
	parameters[11] = resultMat[2][3];

	itkTransform->SetParameters( parameters );
	
	std::cout << "itk parameters: "  << std::endl << itkTransform->GetParameters()  << std::endl;
	std::cout << "itk matrix: "      << std::endl << itkTransform->GetMatrix()      << std::endl;
	std::cout << "itk translation: " << std::endl << itkTransform->GetTranslation() << std::endl;

	itkTransformWriterPointerType transformWriter = itkTransformWriterType::New();
	
	transformWriter->SetInput( itkTransform ); 
	
	if (! strITKTransformFile.empty())
	{
		try
		{
			std::cout << "Writing those parameters to file: " << std::endl << itkTransform->GetParameters() << std::endl;
			
			transformWriter->SetFileName( strITKTransformFile.c_str() );
			transformWriter->Update();
		}
		catch(itk::ExceptionObject e)
		{
			std::cout << "ERROR!!!" << std::endl << e << std::endl;
		}
	}

	/*
	 * Now build the transformation file for the uclToolkit similar, but not the same...
	 */

	std::cout << "NIFTK part: " << std::endl;
	// Bad detour to get an itkAffineTransform with 16 parameters...
	typedef itk::ImageRegistrationFactory<InputImageType, Dimension, double> FactoryType;
	FactoryType::EulerAffineTransformType::Pointer eulerTransform = FactoryType::EulerAffineTransformType::New();
	
	eulerTransform->SetIdentity();

	/*
	 * Confusiong: the itkAffineTransform with 16 parameters (full homogenous matrix...)
	 */
	itkTransformType* uclTransform = eulerTransform->GetFullAffineTransform();
	
	std::cout << "num paramas: " << uclTransform->GetNumberOfParameters() << std::endl;
	parameters = uclTransform->GetParameters();
	
	// first index of the matrix gives the row!
	parameters[ 0] = resultMat[0][0];
	parameters[ 1] = resultMat[0][1];
	parameters[ 2] = resultMat[0][2];
	parameters[ 3] = resultMat[0][3];

	parameters[ 4] = resultMat[1][0];
	parameters[ 5] = resultMat[1][1];
	parameters[ 6] = resultMat[1][2];
	parameters[ 7] = resultMat[1][3];

	parameters[ 8] = resultMat[2][0];
	parameters[ 9] = resultMat[2][1];
	parameters[10] = resultMat[2][2];
	parameters[11] = resultMat[2][3];

	parameters[12] = resultMat[3][0];
	parameters[13] = resultMat[3][1];
	parameters[14] = resultMat[3][2];
	parameters[15] = resultMat[3][3];

	uclTransform->SetParameters( parameters );

	uclTransform->Print( std::cout );

	transformWriter->SetInput( uclTransform ); 
	
	if (! strUCLTransformFile.empty())
	{
		try
		{
			std::cout << "Writing those parameters to file: " << std::endl << itkTransform->GetParameters() << std::endl;
			
			transformWriter->SetFileName( strUCLTransformFile );
			transformWriter->Update();
		}
		catch(itk::ExceptionObject e)
		{
			std::cout << "ERROR!!!" << std::endl << e << std::endl;
		}
	}



	return EXIT_SUCCESS;
}




