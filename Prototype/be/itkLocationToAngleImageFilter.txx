/*=============================================================================

NifTK: An image processing toolkit jointly developed by the
Dementia Research Centre, and the Centre For Medical Image Computing
at University College London.

See:        http://dementia.ion.ucl.ac.uk/
http://cmic.cs.ucl.ac.uk/
http://www.ucl.ac.uk/

Last Changed      : $Date: 2011-09-20 20:57:34 +0100 (Tue, 20 Sep 2011) $
Revision          : $Revision: 7341 $
Last modified by  : $Author: ad $

Original author   : m.clarkson@ucl.ac.uk

Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notices for more information.

============================================================================*/
#ifndef __itkLocationToAngleImageFilter_txx
#define __itkLocationToAngleImageFilter_txx

#include "itkLocationToAngleImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "itkLogHelper.h"
#include "math.h"



namespace itk {

	
	template <class TScalarType, unsigned int NDimensions>
	LocationToAngleImageFilter<TScalarType, NDimensions>
		::LocationToAngleImageFilter()
		: m_dEpsiplon(1.0e-9)
	{
		m_mA  = VNLMatrixType::vnl_matrix( 3, 2 );
		m_mA1 = VNLMatrixType::vnl_matrix( 3, 1 );
		m_mA2 = VNLMatrixType::vnl_matrix( 3, 1 );
		m_mC1 = VNLMatrixType::vnl_matrix( 3, 1 );
		//m_mB  = VNLMatrixType::vnl_matrix( 3, 1 );
		//m_mV  = VNLMatrixType::vnl_matrix( 2, 1 );
	}




	template <class TScalarType, unsigned int NDimensions>
	void
		LocationToAngleImageFilter<TScalarType, NDimensions>
		::PrintSelf(std::ostream& os, Indent indent) const
	{
		Superclass::PrintSelf(os,indent);
	}




	template <class TScalarType, unsigned int NDimensions>
	void
		LocationToAngleImageFilter<TScalarType, NDimensions>
		::BeforeThreadedGenerateData()
	{

		// Check to verify all inputs are specified and have the same metadata, spacing etc...

		const unsigned int numberOfInputs = this->GetNumberOfInputs();

		// We should have exactly 1 inputs.
		if (numberOfInputs != 1)
		{
			itkExceptionMacro(<< "VectorMagnitudeImageFilter should only have 1 input.");
		}

		// Calculate those quantities that can be precalculated

		Vector3Type vA1 = m_UpPoint        - m_BodyAxisPoint1;
		Vector3Type vA2 = m_BodyAxisPoint2 - m_BodyAxisPoint1;

		m_mA1[0][0] = vA1[0];		
		m_mA1[1][0] = vA1[1];		
		m_mA1[2][0] = vA1[2];		
		
		m_mA2[0][0] = vA2[0];		
		m_mA2[1][0] = vA2[1];		
		m_mA2[2][0] = vA2[2];		

		m_mA[0][0]  = vA1[0];
		m_mA[1][0]  = vA1[1];
		m_mA[2][0]  = vA1[2];

		m_mA[0][1]  = vA2[0];
		m_mA[1][1]  = vA2[1];
		m_mA[2][1]  = vA2[2];

		// Angle sign calculations:		
		VNLVectorType vC1 = vnl_cross_3d( vA2.GetVnlVector(), vA1.GetVnlVector() );
		m_mC1[0][0] = vC1[0];
		m_mC1[1][0] = vC1[1];
		m_mC1[2][0] = vC1[2];


#ifdef _DBG
		Matrix3x2Type mTmp1 = Matrix3x2Type( m_mA  );
		std::cout << "A matrix: " << std::endl << mTmp1 << std::endl;
#endif

		m_mAT              = m_mA.transpose();

#ifdef _DBG
		Matrix2x3Type mTmp2 = Matrix2x3Type( m_mAT  );
		std::cout << "A_transpose matrix: " << std::endl << mTmp2 << std::endl;
#endif

		m_mATAinvAT        = vnl_inverse( m_mAT * m_mA ) * m_mAT ;
		
#ifdef _DBG
		Matrix2x3Type mTmp3 = Matrix2x3Type( m_mATAinvAT  );
		std::cout << "inv(AT A) x AT matrix: " << std::endl << mTmp3 << std::endl;
#endif

		VNLMatrixType vTmp = m_mA2.transpose() * m_mA2;
		m_dA2NormSquared   = vTmp[0][0];

#ifdef _DBG
		std::cout << "double A2 vector squared: " << std::endl << m_dA2NormSquared << std::endl;
#endif



	}




	template <class TScalarType, unsigned int NDimensions>
	void
		LocationToAngleImageFilter<TScalarType, NDimensions>
		::ThreadedGenerateData(const InputImageRegionType& outputRegionForThread, int threadNumber) 
	{

		niftkitkDebugMacro(<<"ThreadedGenerateData():Started thread:" << threadNumber);

		// Get Pointers to images.
		typename InputImageType::Pointer inputImage 
			= static_cast< InputImageType * >( this->ProcessObject::GetInput(0) );

		typename OutputImageType::Pointer outputImage 
			= static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

		ImageRegionConstIteratorWithIndex< InputImageType >  inputIterator(inputImage, outputRegionForThread);
		ImageRegionIteratorWithIndex< OutputImageType > outputIterator(outputImage, outputRegionForThread);

		double squaredMagnitude = 0.;

		for (inputIterator.GoToBegin(), outputIterator.GoToBegin(); 
			!inputIterator.IsAtEnd(); 
			++inputIterator, ++outputIterator)
		{
			
			
			PointType p;
			inputImage->TransformIndexToPhysicalPoint( inputIterator.GetIndex(), p );


			// Calculate the relative point B
			VNLMatrixType mB = VNLMatrixType(3,1);

			mB[0][0] = p[0] - m_BodyAxisPoint1[0];
			mB[1][0] = p[1] - m_BodyAxisPoint1[1];
			mB[2][0] = p[2] - m_BodyAxisPoint1[2];

			VNLMatrixType mV = VNLMatrixType(2,1);
			VNLMatrixType mW = VNLMatrixType(1,1);
			mV = m_mATAinvAT * mB;
			mW = ( m_mA2.transpose() * (m_mA * mV) ) / m_dA2NormSquared;

			// finally the angle
			VNLMatrixType mE = mB - (m_mA2 * mW[0][0]);
			VNLMatrixType mF = m_mA * mV - m_mA2 * mW[0][0]; 
			
			double dNormE = mE.array_two_norm();
			double dNormF = mF.array_two_norm();

			VNLMatrixType mExF = mE.transpose() * mF;
			
			// Check direction of the angle to recover values in [-pi ; +ip]
			// idea: when the lenght of the vector from Av to B increases the lenght when added to the corss product of the 
			//       vectors which describe the plane, than the angle is considred positive. 
			double posAngle = -1.0;
			if ( m_mC1.array_two_norm() > (m_mC1  + mE-mF).array_two_norm() ){ posAngle = 1.0; }
			
			double pUpwards;
			(mV[0][0]>0)?(pUpwards=1.0):(pUpwards=-1.0);
			
			OutputPixelType dAlpha = 0.;

			if ( ( dNormE > m_dEpsiplon ) && ( dNormF > m_dEpsiplon ))
			{
				//if ( mV[0][0] > 0. )
				//{
					dAlpha = posAngle * static_cast<OutputPixelType> ( acos( pUpwards * mExF[0][0] / (dNormE * dNormF ) ) );
				//}
				//else
				//{
				//	dAlpha = posAngle * static_cast<OutputPixelType> ( acos( -mExF[0][0] / (dNormE * dNormF ) ) );
				//}
			}

			outputIterator.Set( dAlpha );

#ifdef _DBG
			// Debugging messages when point in the image corner is reached
			if ( ( abs( p[0] + 200.22) < 0.5 ) && ( abs( p[1] + 206.02) < 0.5 ) && ( abs( p[0] + 138.32) < 0.5 ) )
			{
				// Vector v
				Matrix3x1Type dbg_mv = Matrix3x1Type( mV );
				std::cout << "Vector v: " << std::endl<< dbg_mv << std::endl;

				// Vector w
				Matrix1x1Type dbg_mw = Matrix1x1Type( mW );
				std::cout << "Double w: " << std::endl<< dbg_mw << std::endl;


				// Vector E
				Matrix3x1Type dbg_mE = Matrix3x1Type( mE );
				std::cout << "Matrix E: " << std::endl<< dbg_mE << std::endl;
				
				// Vector F
				Matrix3x1Type dbg_mF = Matrix3x1Type( mF );
				std::cout << "Matrix F: " << std::endl<< dbg_mF << std::endl;

				// Vector F
				Matrix3x1Type dbg_mF = Matrix3x1Type( mF );
				std::cout << "Matrix F: " << std::endl<< dbg_mF << std::endl;
			}
#endif

			
		}

		niftkitkDebugMacro(<<"ThreadedGenerateData():Finished thread:" << threadNumber);
	}


} // end namespace

#endif
