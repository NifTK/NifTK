/*
 * ITKppImage2WorldMatrixCalculator.h
 *
 *  Created on: 7 Feb 2011
 *      Author: b-eiben
 */

#ifndef ITKPPIMAGE2WORLDMATRIXCALCULATOR_H_
#define ITKPPIMAGE2WORLDMATRIXCALCULATOR_H_

#include "itkImage.h"

template<typename ImageType, unsigned int Dimension>
class ITKppImage2WorldMatrixCalculator
{
  public:

	typedef double                                                      MatrixElementType;
	typedef itk::Matrix< MatrixElementType, Dimension+1, Dimension+1 >  HomogenousMatrixType;
	typedef typename ImageType::DirectionType                           DirectionType;
	typedef typename ImageType::SizeType                                SizeType;
	typedef typename ImageType::SpacingType                             SpacingType;
	typedef typename ImageType::PointType                               PointType;

	ITKppImage2WorldMatrixCalculator( typename ImageType::Pointer imageIn );
	virtual ~ITKppImage2WorldMatrixCalculator();

	HomogenousMatrixType GetImage2WorldMatrix();
	HomogenousMatrixType GetImage2WolrdTranslationOnlyMatrix();
	HomogenousMatrixType GetImage2WolrdInverseTranslationOnlyMatrix();

  private:
	HomogenousMatrixType m_Image2WorldMatrix;
	HomogenousMatrixType m_Image2WorldTranslationOnlyMatrix;
	HomogenousMatrixType m_Image2WorldInverseTranslationOnlyMatrix;
};

#include "ITKppImage2WorldMatrixCalculator.cpp"

#endif /* ITKPPIMAGE2WORLDMATRIXCALCULATOR_H_ */
