/*
 * itkDisplacementVectorCoordinateAdaptionPixelAccessor.cxx
 *
 *  Created on: 02.12.2010
 *      Author: b-eiben
 */
#ifndef ITKDISPLACEMENTVECTORCOORDINATEADAPTIONPIXELACCESSOR_CXX_
#define ITKDISPLACEMENTVECTORCOORDINATEADAPTIONPIXELACCESSOR_CXX_

#include "itkDisplacementVectorCoordinateAdaptionPixelAccessor.h"


/*
 * Constructor
 */
template< unsigned int Dimension, class VectorComponentType >
itkDisplacementVectorCoordinateAdaptionPixelAccessor< Dimension, VectorComponentType >::
  itkDisplacementVectorCoordinateAdaptionPixelAccessor()
{
	this->m_matrix.SetIdentity();
}

/*
 * Destructor (nothing to do yet)
 */
template< unsigned int Dimension, class VectorComponentType >
itkDisplacementVectorCoordinateAdaptionPixelAccessor< Dimension, VectorComponentType >::
  ~itkDisplacementVectorCoordinateAdaptionPixelAccessor()
{}



template< unsigned int Dimension, class VectorComponentType >
typename itkDisplacementVectorCoordinateAdaptionPixelAccessor<Dimension, VectorComponentType>::ExternalType
itkDisplacementVectorCoordinateAdaptionPixelAccessor<Dimension, VectorComponentType>
  ::Get( const InternalType & vectIn ) const
{
	InternalHomogenousVectorType homVectIn;


	for ( unsigned int i = 0; i < Dimension;  ++i )
	{
		homVectIn[i] = static_cast< MatrixElementType >( vectIn[i] );
	}
	homVectIn[ Dimension ] = 1.0;

	InternalHomogenousVectorType homVectOut = m_matrix * homVectIn;
	// TODO: Check for homVectOut[3] if != 1.0;

	ExternalType vectOut;

	for (unsigned int i = 0;  i < Dimension;  ++i )
	{
		vectOut[ i ] = homVectOut[ i ];
	}

	//vectOut[0] = - vectOut[0];
	//vectOut[1] = - vectOut[1];

	return vectOut;
}




template< unsigned int Dimension, class VectorComponentType >
void
itkDisplacementVectorCoordinateAdaptionPixelAccessor<Dimension, VectorComponentType>
  ::SetHomogenousMatrix( const HomogenousMatrixType & homMatIn )
{
	m_matrix = homMatIn;

	return;
}


#endif /* ITKDISPLACEMENTVECTORCOORDINATEADAPTIONPIXELACCESSOR_CXX_ */
