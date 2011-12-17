/*
 * ITKppImage2WorldMatrixCalculator.cpp
 *
 *  Created on: 7 Feb 2011
 *      Author: b-eiben
 */

#ifndef ITKPPIMAGE2WORLDMATRIXCALCULATOR_CPP_
#define ITKPPIMAGE2WORLDMATRIXCALCULATOR_CPP_

#include "ITKppImage2WorldMatrixCalculator.h"



/**
 * Constructor
 * Calculates the image2world matrix
 */
template< typename ImageType, unsigned int Dimension >
ITKppImage2WorldMatrixCalculator< ImageType, Dimension >::ITKppImage2WorldMatrixCalculator( typename ImageType::Pointer inputImage )
	: m_Image2WorldMatrix(),
	  m_Image2WorldTranslationOnlyMatrix()
{
	HomogenousMatrixType matTMP1;
	HomogenousMatrixType matTMP2;
	HomogenousMatrixType matTMP3;

	m_Image2WorldMatrix.SetIdentity();
	matTMP1.SetIdentity();
	matTMP2.SetIdentity();
	matTMP3.SetIdentity();

	/* Get some basic image information */
	DirectionType direction  = inputImage->GetDirection();
	SizeType      size       = inputImage->GetLargestPossibleRegion().GetSize();
	SpacingType   spacing    = inputImage->GetSpacing();
	PointType     origin     = inputImage->GetOrigin();

	/* Direction... */
	for ( unsigned int i = 0;  i < Dimension;  ++i )
	{
		for ( unsigned int j = 0;  j < Dimension;  ++j )
		{
			m_Image2WorldMatrix(i,j) = direction(i,j);
		}
	}

	/* Image size... */
	matTMP1(0,3) = - (size[0] - 1.0) / 2.0;
	matTMP1(1,3) = - (size[1] - 1.0) / 2.0;
	matTMP1(2,3) = - (size[2] - 1.0) / 2.0;

	/* Spacing... */
	matTMP2(0,0) = spacing[0];
	matTMP2(1,1) = spacing[1];
	matTMP2(2,2) = spacing[2];

	/* origin */
	matTMP3(0,3) = origin[0];
	matTMP3(1,3) = origin[1];
	matTMP3(2,3) = origin[2];

	m_Image2WorldMatrix = matTMP3 * m_Image2WorldMatrix * matTMP2 * matTMP1;

	m_Image2WorldTranslationOnlyMatrix.SetIdentity();
	m_Image2WorldTranslationOnlyMatrix(0,3) = m_Image2WorldMatrix(0,3);
	m_Image2WorldTranslationOnlyMatrix(1,3) = m_Image2WorldMatrix(1,3);
	m_Image2WorldTranslationOnlyMatrix(2,3) = m_Image2WorldMatrix(2,3);

	m_Image2WorldInverseTranslationOnlyMatrix.SetIdentity();
	m_Image2WorldInverseTranslationOnlyMatrix(0,3) = - m_Image2WorldMatrix(0,3);
	m_Image2WorldInverseTranslationOnlyMatrix(1,3) = - m_Image2WorldMatrix(1,3);
	m_Image2WorldInverseTranslationOnlyMatrix(2,3) = - m_Image2WorldMatrix(2,3);
}


/**
 * Gets the image to world matrix
 */
template< typename ImageType, unsigned int Dimension >
typename ITKppImage2WorldMatrixCalculator< ImageType, Dimension >::HomogenousMatrixType
ITKppImage2WorldMatrixCalculator< ImageType, Dimension >::GetImage2WorldMatrix()
{
	return m_Image2WorldMatrix;
}



/**
 * Gets the image to world matrix translation only matrix
 */
template< typename ImageType, unsigned int Dimension >
typename ITKppImage2WorldMatrixCalculator< ImageType, Dimension >::HomogenousMatrixType
ITKppImage2WorldMatrixCalculator< ImageType, Dimension >::GetImage2WolrdTranslationOnlyMatrix()
{
	return m_Image2WorldTranslationOnlyMatrix;
}




/**
 * Gets the image to world matrix inverse translation only matrix
 */
template< typename ImageType, unsigned int Dimension >
typename ITKppImage2WorldMatrixCalculator< ImageType, Dimension >::HomogenousMatrixType
ITKppImage2WorldMatrixCalculator< ImageType, Dimension >::GetImage2WolrdInverseTranslationOnlyMatrix()
{
	return m_Image2WorldInverseTranslationOnlyMatrix;
}




/**
 * Destructor, nothing to do yet...
 */
template< typename ImageType, unsigned int Dimension >
ITKppImage2WorldMatrixCalculator< ImageType, Dimension >::~ITKppImage2WorldMatrixCalculator()
{
	// Nothing to do yet...
}

#endif  // ITKPPIMAGE2WORLDMATRIXCALCULATOR_CPP_
