/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.

 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: mjc                 $
 $Date:: 2010-08-11 08:28:23 +#$
 $Rev:: 3647                   $

 Copyright (c) UCL : See the file LICENSE.txt in the top level
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "ConversionUtils.h"
#include "CommandLineParser.h"
#include "FileHelper.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImage.h"
#include "itkImageRegionIterator.h"

#include "itkEuler3DTransform.h"
#include "itkVectorResampleImageFilter.h"

#include "itkScalarImageToHistogramGenerator.h"
#include "itkVectorLinearInterpolateImageFunction.h"
#include "itkVectorNearestNeighborInterpolateImageFunction.h"

#include "itkAffineTransform.h"
#include "itkTransformToDeformationFieldSource.h"
#include "itkMatrix.h"
#include "vnl/vnl_matrix.h"

#include "itkTransformFileWriter.h"

#include "itkMetaDataDictionary.h"

#include <vector>
#include <algorithm>
#include <math.h>
#include <boost/filesystem.hpp>

namespace bfs = boost::filesystem;


struct niftk::CommandLineArgumentDescription clArgList[] =
{
    { OPT_SWITCH, "f",                         0,          "Forward interpretation of field 1 (use only if you know what you are doing)." },
    { OPT_SWITCH, "mat",                       0,          "Second input file is a 4x4 matrix which will internally be converted into a deformation field." },
    { OPT_SWITCH, "matRot",                    0,          "Calculate the matrix which was produced in a rotated coordinate system: M' = R M R" },
    { OPT_STRING, "matField",                  "filename", "Write the displacement vector field from the matrix to the file specified." },
    { OPT_INT,    "mvalue",                    "value",    "The mask intensity used to determine the region of interest." },
    { OPT_STRING, "mask",                      "filename", "Calculate error only over mask region." },
    { OPT_STRING, "oi",                        "filename", "Output an image of the error magnitude at each voxel." },
    { OPT_STRING, "oivect",                    "filename", "Output an image of the error vector at each voxel." },
    { OPT_STRING, "st",                        "filename", "File name to which the statistics will be written (plain text)." },
    { OPT_DOUBLE, "p",                         "value",    "Calculate the p-th percentile of the error magnitude. [95.0]" },
	{ OPT_SWITCH, "vox1",                      0,          "Deformation field 1 gives the deformation in voxels."  },
	{ OPT_SWITCH, "vox2",                      0,          "Deformation field 2 gives the deformation in voxels."  },
    { OPT_STRING | OPT_LONELY | OPT_REQ, NULL, "filename", "Input deformation field 1." },
    { OPT_STRING | OPT_LONELY | OPT_REQ, NULL, "filename", "Input deformation field 2." },
    { OPT_DONE, NULL, NULL,                                "Calculates the error between two deformation fields which act in the same direction "
                                                           "(flag -f assumes different directions of the deformations fields F1: forward/putting, "
                                                           "F2: backward/getting). "
                                                           "Error calculated at voxel positions of mask or deformation field 1." }
};



enum
{
    O_FIELD1FWD_FIELD2BKWD = 0,
    O_MATRIX_INSTEAD_OF_FIELD,
    O_MATRIX_CALCULATED_IN_ROTATED_SYSTEM,
    O_MATRIX_FIELD_NAME,
    O_MASK_VALUE,
    O_MASK,
    O_OUTPUT_IMAGE,
    O_OUTPUT_VECT_IMAGE,
    O_STAT_NAME,
    O_PERCENTILE,
	O_VOXEL_SPACING_FIELD1,
	O_VOXEL_SPACING_FIELD2,
    O_DEFORMATION_1,
    O_DEFORMATION_2,
};



int main(int argc, char ** argv)
{
    bool   bField1ForwardDisplacement;
    bool   b2ndFieldByMatrix;
    bool   bMatrixFromRotatedSystem;
    double dPercentile = 95.0;

//	bool bVoxelSpacingField1 = FALSE;
//	bool bVoxelSpacingField2 = FALSE;
	bool bVoxelSpacingField1 = false;
	bool bVoxelSpacingField2 = false;

    /*
     * Default filenames
     */
    char *in1Name           = NULL;
    char *in2Name           = NULL;
    char *maskName          = NULL;
    char *outName           = NULL;
    char *outVectName       = NULL;
    char *pcHistoName       = NULL;
    char *pcMatrixFiledName = NULL;

    const unsigned int Dimension = 3;

    typedef int MaskPixelType;
    MaskPixelType mask_value = 1;

    /*
     * Image and reader definitions
     */
    typedef float InternalPixelType;
    typedef itk::Vector<InternalPixelType, Dimension> VectorPixelType;
    typedef itk::Image<VectorPixelType, Dimension> FieldType;

    typedef FieldType::Pointer FieldPointer;
    typedef itk::ImageRegionIterator<FieldType> FieldIterator;
    typedef FieldType::PixelType DisplacementType;

    typedef FieldType::IndexType FieldIndexType;
    typedef FieldType::RegionType FieldRegionType;
    typedef FieldType::SizeType FieldSizeType;
    typedef FieldType::SpacingType FieldSpacingType;
    typedef FieldType::PointType FieldPointType;
    typedef FieldType::DirectionType FieldDirectionType;

    typedef itk::ImageFileReader<FieldType> FieldReaderType;

    typedef itk::Image< MaskPixelType, Dimension >     MaskImageType;
    typedef itk::ImageRegionIterator< MaskImageType >  MaskIterator;
    typedef itk::ImageFileReader    < MaskImageType >  MaskReaderType;

    typedef itk::Image< InternalPixelType, Dimension > InternalImageType;

    typedef itk::ImageFileWriter    < InternalImageType >  InternalImageWriterType;
    typedef itk::ImageRegionIterator< InternalImageType >  ImageIterator;

    typedef itk::ImageFileWriter    < FieldType >  FieldWriterType;
    typedef itk::ImageRegionIterator< FieldType >  FieldIteratorType;

    /*
     * Parse the command line
     */
    niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

    CommandLineOptions.GetArgument( O_FIELD1FWD_FIELD2BKWD,                bField1ForwardDisplacement );
    CommandLineOptions.GetArgument( O_MATRIX_INSTEAD_OF_FIELD,             b2ndFieldByMatrix          );
    CommandLineOptions.GetArgument( O_MATRIX_CALCULATED_IN_ROTATED_SYSTEM, bMatrixFromRotatedSystem   );
    CommandLineOptions.GetArgument( O_MATRIX_FIELD_NAME,                   pcMatrixFiledName          );

    CommandLineOptions.GetArgument( O_MASK_VALUE, mask_value );
    CommandLineOptions.GetArgument( O_MASK,       maskName   );

    CommandLineOptions.GetArgument( O_OUTPUT_IMAGE,      outName     );
    CommandLineOptions.GetArgument( O_OUTPUT_VECT_IMAGE, outVectName );

    CommandLineOptions.GetArgument( O_DEFORMATION_1, in1Name );
    CommandLineOptions.GetArgument( O_DEFORMATION_2, in2Name );

    CommandLineOptions.GetArgument( O_STAT_NAME, pcHistoName );
    CommandLineOptions.GetArgument( O_PERCENTILE, dPercentile );

	CommandLineOptions.GetArgument( O_VOXEL_SPACING_FIELD1, bVoxelSpacingField1 );
	CommandLineOptions.GetArgument( O_VOXEL_SPACING_FIELD2, bVoxelSpacingField2 );

	if ( bVoxelSpacingField1 )
		std::cout << "Using voxel spacing as scaling for field 1!" << std::endl;
	if ( bVoxelSpacingField2 )
		std::cout << "Using voxel spacing as scaling for field 2!" << std::endl;


    /*
     * Create the image readers etc.
     */
    FieldReaderType::Pointer fieldReader1 = FieldReaderType::New();
    FieldReaderType::Pointer fieldReader2 = FieldReaderType::New();

    FieldType::Pointer singleField1;
    FieldType::Pointer singleField2;

    MaskReaderType::Pointer maskReader = MaskReaderType::New();

    MaskImageType::Pointer mask;



    /*
     * Error images
     */
    InternalImageType::Pointer errorImage;
    InternalImageType::Pointer initialerrorImage;
    FieldType::Pointer         errorVectorImage;

    // Prepare resampling of deformation field 2 if needed
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    typedef itk::VectorResampleImageFilter<FieldType, FieldType> FieldResampleFilterType;

    FieldResampleFilterType::Pointer fieldResample = FieldResampleFilterType::New();

    VectorPixelType zeroDisplacement;
    zeroDisplacement.Fill( 0.0f );

    typedef itk::Euler3DTransform< double > RigidTransformType;

    RigidTransformType::Pointer rigidIdentityTransform = RigidTransformType::New();
    rigidIdentityTransform->SetIdentity();

    // Read the deformation fields
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fieldReader1->SetFileName( in1Name );
    try
    {
        fieldReader1->Update();
    }
    catch (itk::ExceptionObject & excp)
    {
        std::cerr << "Exception thrown " << std::endl;
        std::cerr << excp << std::endl;
    }

    singleField1 = fieldReader1->GetOutput();

    /*
     * Either read deformation field or generate one from a given matrix
     */
    if ( ! b2ndFieldByMatrix )
    {
		fieldReader2->SetFileName( in2Name );

		try
		{
			fieldReader2->Update();
		}
		catch ( itk::ExceptionObject & excp )
		{
			std::cerr << "Exception thrown " << std::endl;
			std::cerr << excp << std::endl;
		}

		singleField2 = fieldReader2->GetOutput();
    }
    else
    {
		/*
		 * Try to create a deformation field from an affine matrix (text file)...
		 */
		typedef itk::AffineTransform< InternalPixelType, Dimension >   AffineTransformType;
		typedef AffineTransformType::Pointer                           AffineTransformPointerType;
		typedef AffineTransformType::ParametersType                    AffineParametersType;


		AffineTransformPointerType affineTransform = AffineTransformType::New();
		affineTransform->SetIdentity();

		AffineParametersType affParams = affineTransform->GetParameters();


		/* Set up the read matrix (mat), the rotation matrix (rot) and the result matrix (res) */
		typedef vnl_matrix<double> VnlMatrixType;
		VnlMatrixType mat(4,4);
		VnlMatrixType res(4,4);
		VnlMatrixType rot(4,4);

		/* fill the rotation -- 180 degrees around the z-axis */
		rot.set_identity();
		rot(0,0) = -1.;
		rot(1,1) = -1.;

		/*
		 * Read the affine text file directly into the transformation parameters...
		 */
		std::ifstream affineFile;
		affineFile.open( in2Name );

		if( affineFile.is_open() )
		{
			float value1, value2, value3, value4;

			for ( unsigned int i = 0;  i < Dimension+1;  ++i )
			{
				affineFile >> value1 >> value2 >> value3 >> value4;

				mat(i,0) = value1;
				mat(i,1) = value2;
				mat(i,2) = value3;
				mat(i,3) = value4;
			}

			/* calculate the result */
			if ( bMatrixFromRotatedSystem )  res = rot * mat * rot;
			else                             res = mat;

			/* feed the translated transformation back into the transformation parameters */
			affParams[ 0 ] = res( 0, 0 );   affParams[ 1 ] = res( 0, 1 );   affParams[ 2 ] = res( 0, 2 );      affParams[  9 ] = res( 0, 3 );
			affParams[ 3 ] = res( 1, 0 );   affParams[ 4 ] = res( 1, 1 );   affParams[ 5 ] = res( 1, 2 );      affParams[ 10 ] = res( 1, 3 );
			affParams[ 6 ] = res( 2, 0 );   affParams[ 7 ] = res( 2, 1 );   affParams[ 8 ] = res( 2, 2 );      affParams[ 11 ] = res( 2, 3 );

			/*
			 * Some debug output
			 */
			std::cout << " -> mat:" << std::endl << mat
					  << " -> rot:" << std::endl << rot
					  << " -> res:" << std::endl << res;
			affineFile.close();
		}
		else
		{
			std::cout << "Could not read the matrix file!" << std::endl;
			return EXIT_FAILURE;
		}


		affineTransform->SetParameters( affParams );

		/*
		 * Generate the deformation field from the transformation...
		 */
		typedef itk::TransformToDeformationFieldSource< FieldType, InternalPixelType >  DeformationFieldGeneratorType;
		DeformationFieldGeneratorType::Pointer defGenerator = DeformationFieldGeneratorType::New();

		defGenerator->SetOutputParametersFromImage( singleField1    );
		defGenerator->SetTransform                ( affineTransform );

		try
		{
			defGenerator->Update();
		}
		catch (itk::ExceptionObject e)
		{
			std::cout << e << std::endl;
			return EXIT_FAILURE;
		}

		singleField2 = defGenerator->GetOutput();

		if (pcMatrixFiledName != NULL)
		{

			FieldWriterType::Pointer fieldWriter = FieldWriterType::New();
			fieldWriter->SetInput   ( singleField2      );
			fieldWriter->SetFileName( pcMatrixFiledName );
			try
			{
				fieldWriter->Update();
			}
			catch (itk::ExceptionObject e)
			{

				std::cerr << "Exception caught: " << std::endl << std::endl;
				return EXIT_FAILURE;
			}
		}
    }


    FieldRegionType    region    = singleField1->GetLargestPossibleRegion();
    FieldSizeType      size      = singleField1->GetLargestPossibleRegion().GetSize();
    FieldPointType     origin    = singleField1->GetOrigin();
    FieldSpacingType   spacing   = singleField1->GetSpacing();
    FieldDirectionType direction = singleField1->GetDirection();

    // Read the mask image
    // ~~~~~~~~~~~~~~~~~~~

    if (maskName != NULL)
    {
        maskReader->SetFileName(maskName);
        try
        {
            maskReader->Update();
        } catch (itk::ExceptionObject & excp)
        {
            std::cerr << "Exception thrown " << std::endl;
            std::cerr << excp << std::endl;
        }
        mask = maskReader->GetOutput();

        FieldRegionType    regionM    = mask->GetLargestPossibleRegion();
        FieldSizeType      sizeM      = mask->GetLargestPossibleRegion().GetSize();
        FieldPointType     originM    = mask->GetOrigin();
        FieldSpacingType   spacingM   = mask->GetSpacing();
        FieldDirectionType directionM = mask->GetDirection();

        bool doResampleField = false;

        for (unsigned int i = 0; i < Dimension; i++)
        {
            if (spacingM[i] != spacing[i])
            {
                doResampleField = true;
                std::cout << "Resampling field 1 relative to mask, spacing ["
                        << niftk::ConvertToString((int) i) << "] "
                        << niftk::ConvertToString(spacing[i]) << " to "
                        << niftk::ConvertToString(spacingM[i]);
            }
            if (sizeM[i] != size[i])
            {
                doResampleField = true;
                std::cout << "Resampling field 1 relative to mask, size ["
                        << niftk::ConvertToString((int) i) << "] "
                        << niftk::ConvertToString(size[i]) << " to "
                        << niftk::ConvertToString(sizeM[i]);
            }
            if (originM[i] != origin[i])
            {
                doResampleField = true;
                std::cout << "Resampling field 1 relative to mask, origin ["
                        << niftk::ConvertToString((int) i) << "] "
                        << niftk::ConvertToString(origin[i]) << " to "
                        << niftk::ConvertToString(originM[i]);
            }
            for (unsigned int j = 0; j < Dimension; j++)
            {
                if (directionM[i][j] != direction[i][j])
                {
                    doResampleField = true;
                    std::cout << "Resampling field 1 relative to mask, direction ["
                            << niftk::ConvertToString((int) i) << "]["
                            << niftk::ConvertToString((int) j) << "] "
                            << niftk::ConvertToString(direction[i][j]) << " to "
                            << niftk::ConvertToString(directionM[i][j]);
                }
            }

        }
        if ( doResampleField )
        {
            std::cout << "Changing field 1 to image format size and spacing of mask";
            // resample if necessary
            fieldResample->SetSize(sizeM);
            fieldResample->SetOutputOrigin(originM);
            fieldResample->SetOutputSpacing(spacingM);
            fieldResample->SetOutputDirection(directionM);
            fieldResample->SetDefaultPixelValue(zeroDisplacement);
            fieldResample->SetTransform(rigidIdentityTransform);

            fieldResample->SetInput(singleField1);
            fieldResample->Update();
            singleField1 = fieldResample->GetOutput();
            singleField1->DisconnectPipeline();

            region = regionM;
            size = sizeM;
            origin = originM;
            spacing = spacingM;
            direction = directionM;
        }
    }

    // Prepare error images
    // ~~~~~~~~~~~~~~~~~~~~

    initialerrorImage = InternalImageType::New();
    initialerrorImage->SetRegions  ( region    );
    initialerrorImage->SetOrigin   ( origin    );
    initialerrorImage->SetSpacing  ( spacing   );
    initialerrorImage->SetDirection( direction );
    initialerrorImage->Allocate();

    errorImage = InternalImageType::New();
    errorImage->SetRegions  ( region    );
    errorImage->SetOrigin   ( origin    );
    errorImage->SetSpacing  ( spacing   );
    errorImage->SetDirection( direction );
    errorImage->Allocate();

    errorVectorImage = FieldType::New();
    errorVectorImage->SetRegions  ( region    );
    errorVectorImage->SetOrigin   ( origin    );
    errorVectorImage->SetSpacing  ( spacing   );
    errorVectorImage->SetDirection( direction );
    errorVectorImage->Allocate();

    // Prepare resampling of deformation field 2 if needed
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fieldResample->SetSize             ( size             );
    fieldResample->SetOutputOrigin     ( origin           );
    fieldResample->SetOutputSpacing    ( spacing          );
    fieldResample->SetOutputDirection  ( direction        );
    fieldResample->SetDefaultPixelValue( zeroDisplacement );

    // check if resampling is necessary

    FieldRegionType    region2    = singleField2->GetLargestPossibleRegion();
    FieldSizeType      size2      = singleField2->GetLargestPossibleRegion().GetSize();
    FieldPointType     origin2    = singleField2->GetOrigin();
    FieldSpacingType   spacing2   = singleField2->GetSpacing();
    FieldDirectionType direction2 = singleField2->GetDirection();

    bool doResampleField = false;

    for ( unsigned int i = 0;  i < Dimension;  i++ )
    {
        if ( spacing2[i] != spacing[i] )
        {
            doResampleField = true;
            std::cout << "Resampling field 2 relative to field 1, spacing ["
                    << niftk::ConvertToString((int) i) << "] "
                    << niftk::ConvertToString(spacing2[i]) << " to "
                    << niftk::ConvertToString(spacing[i]);
        }
        if ( size2[i] != size[i] )
        {
            doResampleField = true;
            std::cout << "Resampling field 2 relative to field 1, size ["
                    << niftk::ConvertToString((int) i) << "] "
                    << niftk::ConvertToString(size2[i]) << " to "
                    << niftk::ConvertToString(size[i]);
        }
        if ( origin2[i] != origin[i] )
        {
            doResampleField = true;
            std::cout << "Resampling field 2 relative to field 1, origin ["
                    << niftk::ConvertToString((int) i) << "] "
                    << niftk::ConvertToString(origin2[i]) << " to "
                    << niftk::ConvertToString(origin[i]);
        }
        for (unsigned int j = 0; j < Dimension; j++)
        {
            if (direction2[i][j] != direction[i][j])
            {
                doResampleField = true;
                std::cout << "Resampling field 2 relative to field 1, direction ["
                        << niftk::ConvertToString((int) i) << "]["
                        << niftk::ConvertToString((int) j) << "] "
                        << niftk::ConvertToString(direction2[i][j]) << " to "
                        << niftk::ConvertToString(direction[i][j]);
            }
        }
    }

    if ( doResampleField )
    {
        std::cout << "Changing field 2 to image format size and "
                                       "spacing of deformation field 1";

        // resample if necessary
        fieldResample->SetTransform( rigidIdentityTransform );
        fieldResample->SetInput    ( singleField2           );
        fieldResample->Update();
        singleField2 = fieldResample->GetOutput();
        singleField2->DisconnectPipeline();
    }


    // Iterate through computing error
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    InternalPixelType sumDiffSq = 0;

    FieldIndexType Findex;

    DisplacementType displacement1;
    DisplacementType displacement2;
    DisplacementType difference;

    InternalPixelType error;
    InternalPixelType meanError = 0.0;
    InternalPixelType maxError  = 0.0;
    InternalPixelType minError  = 1000000.0;
    InternalPixelType stdError  = 0.0;

    InternalPixelType initialerror;
    InternalPixelType initialmeanError = 0.0;
    InternalPixelType initialmaxError  = 0.0;
    InternalPixelType initialminError  = 1000000.0;
    InternalPixelType initialstdError  = 0.0;

    FieldIterator itField1( singleField1, singleField1->GetLargestPossibleRegion() );
    FieldIterator itField2( singleField2, singleField2->GetLargestPossibleRegion() );

    unsigned int N = 0;
    // TODO consider this being double
    std::vector< InternalPixelType > errorList;
    std::vector< InternalPixelType > initialErrorList;

    std::vector< InternalPixelType >::iterator errorListIterator;
    std::vector< InternalPixelType >::iterator initialErrorListIterator;

	// Get the latest spacing
	FieldSpacingType voxSpacing1 = singleField1->GetSpacing();
	FieldSpacingType voxSpacing2 = singleField2->GetSpacing();


    /*
     * Different treatments for
     *   a) displacement fields in the same direction (usually both backward) and
     *   b) displacement fields in different directions (assuming forward displacement
     *      for field 1 and backward for field 2)
     */
    if ( bField1ForwardDisplacement == false )
    {
        for ( itField1.Begin(), itField2.Begin();  !itField1.IsAtEnd();  ++itField1, ++itField2 )
        {
            Findex = itField1.GetIndex();

            if ( (maskName == NULL) || (mask->GetPixel( Findex ) == mask_value) )
            {
                // displacement
                displacement1 = itField1.Get();
                displacement2 = itField2.Get();

				if (bVoxelSpacingField1)
				{
					displacement1[0] = displacement1[0] * voxSpacing1[0];
					displacement1[1] = displacement1[1] * voxSpacing1[1];
					displacement1[2] = displacement1[2] * voxSpacing1[2];
				}
				if (bVoxelSpacingField2)
				{
					displacement2[0] = displacement2[0] * voxSpacing2[0];
					displacement2[1] = displacement2[1] * voxSpacing2[1];
					displacement2[2] = displacement2[2] * voxSpacing2[2];
				}

                difference = displacement1 - displacement2;

                initialerror = displacement1.GetNorm();
                error        = difference.GetNorm();

                initialmeanError = initialmeanError + initialerror;
                meanError        = meanError        + error;

                if (initialmaxError < initialerror)
                    initialmaxError = initialerror;
                if (initialminError > initialerror)
                    initialminError = initialerror;

                if ( maxError < error )
                     maxError = error;
                if ( minError > error )
                     minError = error;

                N++;

                // always store in image
                initialerrorImage->SetPixel( Findex, initialerror );
                errorImage       ->SetPixel( Findex, error        );
                errorVectorImage ->SetPixel( Findex, difference   );

                errorList.push_back( error );
                initialErrorList.push_back( initialerror );
            }
        }
    }
    else /* special treatment added for Tanner simulations... */
    {
        /*
         * Prepare the usage of the vector interpolator, as the position in the second (backward) field needs to be interpolated
         */
        typedef itk::VectorLinearInterpolateImageFunction< FieldType, InternalPixelType >  VectorInterpolatorType;
        typedef VectorInterpolatorType::Pointer                                            VectorInterpolatorPointerType;

        VectorInterpolatorPointerType field2Interpolator = VectorInterpolatorType::New();
        field2Interpolator->SetInputImage( singleField2 );

        /*
         * Go through field1 and evaluate the composition:
         *   field2 o field1(x)
         */
        for ( itField1.Begin();  !itField1.IsAtEnd();  ++itField1 )
        {
            Findex = itField1.GetIndex();

            if ( (maskName == NULL) || (mask->GetPixel( Findex ) == mask_value) )
            {
                FieldPointType point1;
                FieldPointType point1forward;

                /* Transform the index of the iterator to the physical image point... */
                singleField1->TransformIndexToPhysicalPoint( Findex, point1 );

                /* Get the displacement vector */
                VectorPixelType displacement1 = itField1.Get();
				if (bVoxelSpacingField1)
				{
					displacement1[0] = displacement1[0] * voxSpacing1[0];
					displacement1[1] = displacement1[1] * voxSpacing1[1];
					displacement1[2] = displacement1[2] * voxSpacing1[2];
				}

                /* Add the displacement to the point as field1 is describes the deformation in a forward manner... */
                point1forward = point1 + displacement1;

                /* Evaluate the backward deformation  */
                if ( field2Interpolator->IsInsideBuffer( point1forward ) )
                {
                    VectorPixelType displacement2 = field2Interpolator->Evaluate( point1forward );
					
					if (bVoxelSpacingField2)
					{
						displacement2[0] = displacement2[0] * voxSpacing2[0];
						displacement2[1] = displacement2[1] * voxSpacing2[1];
						displacement2[2] = displacement2[2] * voxSpacing2[2];
					}

                    /* in an ideal world the resulting points are exactly the same... */
                    VectorPixelType difVect = displacement1 + displacement2;

                    initialerror = displacement1.GetNorm();
                    error        = difVect.GetNorm();

                    initialmeanError = initialmeanError + initialerror;
                    meanError        = meanError        + error;

                    if ( initialmaxError < initialerror )  initialmaxError = initialerror;
                    if ( initialminError > initialerror )  initialminError = initialerror;
                    if ( maxError        < error        )  maxError        = error;
                    if ( minError        > error        )  minError        = error;

                    N++;

                    // always store in image
                    initialerrorImage->SetPixel( Findex, initialerror );
                    errorImage       ->SetPixel( Findex, error        );
                    errorVectorImage ->SetPixel( Findex, difVect      );

                    errorList.push_back( error );
                    initialErrorList.push_back( initialerror );
                }
            }
        }
    }

    if ( N <= 1 )
    {
        std::cout << "ERROR: Not enough measurements taken!" << std::endl;
        return EXIT_FAILURE;
    }

    /*
     * Calculate linear interpolation of the percentile
     */
    std::sort( errorList.begin(),        errorList.end()        );
    std::sort( initialErrorList.begin(), initialErrorList.end() );

    /* note: the rank is calculated from (1 .. N) and the index of errorList runs from (0 .. N-1) */
    int iRank = static_cast< int >( niftk::Round( dPercentile * N / 100.0 + 0.5 ) );

    double dErrorValPercentileLin;
    double dInitialErrorValPercentileLin;

    if ( iRank <= 1 )
    {
        dErrorValPercentileLin        = errorList       [ 0 ];
        dInitialErrorValPercentileLin = initialErrorList[ 0 ];
    }
    else if ( static_cast<unsigned int>( iRank ) >= N )
    {
        dErrorValPercentileLin        = errorList       [ N-1 ];
        dInitialErrorValPercentileLin = initialErrorList[ N-1 ];
    }
    else
    {
        /* lower percentile */
        double dLower = 100.0 / N * (iRank - 0.5 );

        dErrorValPercentileLin        = errorList       [ iRank - 1 ] + ( dPercentile - dLower ) * ( errorList       [ iRank ] - errorList       [ iRank - 1 ] ) * N / 100.;
        dInitialErrorValPercentileLin = initialErrorList[ iRank - 1 ] + ( dPercentile - dLower ) * ( initialErrorList[ iRank ] - initialErrorList[ iRank - 1 ] ) * N / 100.;
    }

    std::cout << "linear interpolated percentile: " << dErrorValPercentileLin << std::endl;

    initialmeanError = initialmeanError / N;
    meanError        = meanError        / N;


    /*
     * second pass, compute standard deviation
     */

    /* of the initial error */
    sumDiffSq = 0;

    for ( initialErrorListIterator =  initialErrorList.begin();
          initialErrorListIterator != initialErrorList.end();
          ++initialErrorListIterator )
    {
        initialerror = *initialErrorListIterator;
        sumDiffSq = sumDiffSq +   (initialerror - initialmeanError)
                                * (initialerror - initialmeanError);
    }

    initialstdError = vcl_sqrt( sumDiffSq / (N - 1) );

    std::cout << "Initial min  error: "  << niftk::ConvertToString( initialminError  );
    std::cout << "Initial mean error: "  << niftk::ConvertToString( initialmeanError );
    std::cout << "Initial max  error: "  << niftk::ConvertToString( initialmaxError  );
    std::cout << "Initial std  error: "  << niftk::ConvertToString( initialstdError  );

    /* of the error */
    sumDiffSq = 0;

    for ( errorListIterator  = errorList.begin();
          errorListIterator != errorList.end();
          ++errorListIterator )
    {
        error     = *errorListIterator;
        sumDiffSq = sumDiffSq + (error - meanError) * (error - meanError);
    }

    stdError = vcl_sqrt(sumDiffSq / (N - 1));

    std::cout << "Registration min  error: "  << niftk::ConvertToString( minError  );
    std::cout << "Registration mean error: "  << niftk::ConvertToString( meanError );
    std::cout << "Registration max  error: "  << niftk::ConvertToString( maxError  );
    std::cout << "Registration std  error: "  << niftk::ConvertToString( stdError  );

    // Write the error image to a file
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if (outName != NULL)
    {
        InternalImageWriterType::Pointer writer = InternalImageWriterType::New();

        writer->SetFileName( outName    );
        writer->SetInput   ( errorImage );

        std::cout <<  "Writing error image to file: "  << outName;

        try
        {
            writer->Update();
        }
        catch (itk::ExceptionObject & err)
        {
            std::cerr << "ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
            return EXIT_FAILURE;
        }
    }


    /*
     * Save output vector image
     */
    if (outVectName != NULL)
    {
        FieldWriterType::Pointer fieldWriter = FieldWriterType::New();

        fieldWriter->SetInput   ( errorVectorImage );
        fieldWriter->SetFileName( outVectName      );

        std::cout <<  "Writing error vector image to file: "  << outName;

        try
        {
            fieldWriter->Update();
        }
        catch ( itk::ExceptionObject err )
        {
            std::cerr << "ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
            return EXIT_FAILURE;
        }
    }

    /*
     * Generate the histogram for statistical evaluation
     */
    if (pcHistoName != NULL)
    {
        std::cout << "Generating statistics text file..." << std::endl;

		bfs::path pthField1( in1Name );
		bfs::path pthField2( in2Name );
		

        /*
         * Write the result to a text file
         */
        FILE* fpOutfile = NULL;
        fpOutfile = fopen( pcHistoName, "a" );

        if(fpOutfile == NULL)
        {
        	perror("Failed to open file for appending");
        	return EXIT_FAILURE;
        }

        if (!niftk::FileExists(pcHistoName) || niftk::FileIsEmpty(pcHistoName))
        {
            char* pcLine = (char*) "--------------" ;
            fprintf( fpOutfile, "%-14.14s  %-14.14s  %-14.14s  %-14.14s  %-14.14s  "
				                "%-14.14s  %-14.14s  %-14.14s  %-14.14s  %-14.14s  "
								"%-14.14s  %-30.26s  %-60.56s\n",
                                "initial_min", "initial_mean", "initial_max", "initial_std", "initial_perc.",
                                "error_min",   "error_mean",   "error_max",   "error_std",   "error_perc.",  
								"percentile",  "field1", "field2");

            fprintf( fpOutfile, "%-14.14s  %-14.14s  %-14.14s  %-14.14s  %-14.14s  "
				                "%-14.14s  %-14.14s  %-14.14s  %-14.14s  %-14.14s  "
								"%-14.14s  %-30.26s  %-60.56s\n",
								pcLine, pcLine, pcLine, pcLine, pcLine,
								pcLine, pcLine, pcLine, pcLine, pcLine,
								pcLine, pcLine, pcLine );
        }


        fprintf( fpOutfile, "%-14.10f  %-14.10f  %-14.10f  %-14.10f  %-14.10f  "
				            "%-14.10f  %-14.10f  %-14.10f  %-14.10f  %-14.10f  "
							"%-14.10f  %-30.26s  %-60.56s\n",
							initialminError, initialmeanError, initialmaxError, initialstdError, dInitialErrorValPercentileLin,
							minError,        meanError,        maxError,        stdError,        dErrorValPercentileLin,        
							dPercentile, pthField1.filename().c_str(), pthField2.filename().c_str() );

        fclose( fpOutfile );
    }

    return EXIT_SUCCESS;
}

