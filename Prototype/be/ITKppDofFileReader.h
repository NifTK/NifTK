/*
 * ITKppDofFileReader.h
 *
 *  Created on: 4 Feb 2011
 *      Author: b-eiben
 */

#ifndef ITKPPDOFFILEREADER_H_
#define ITKPPDOFFILEREADER_H_

#include <string>
#include <iostream>
#include <fstream>
#include "itkMatrix.h"

#include "FileHelper.h"
#include "math.h"

struct DofFileHeader
{
	unsigned int uiMagicNumber;
	unsigned int uiTransformationType;
	unsigned int uiNumberOfDofs;
};




/*
 * This class implements a reader for the dof files generated with itk++
 * Currently only the rigid and the affine dof files (3D) are supported
 */
template <unsigned int Dimension = 3>
class ITKppDofFileReader
{
  public:
	typedef itk::Matrix< double, 4, 4 > HomogenousMatrixType;

	ITKppDofFileReader( std::string dofFileName );
	virtual ~ITKppDofFileReader();

	void PrintDofs();
	void PrintHomogenousMatrix();

	HomogenousMatrixType GetHomogenousMatrix();
	HomogenousMatrixType GetTranslationalPart();

  private:
	/* Functions */
	void swapFileHeaderBytes();
	void swapDofBytes();
	void swap32( char* pcIn );
	void swap64( char* pcIn );
	void CalculateHomogenousMatrix();

	/* Variables */
	double *             m_pdDofs;
	const static double  m_dPI;
	DofFileHeader*       m_dofsHeader;
	bool                 m_bSwapped;
	HomogenousMatrixType m_homogenousMatrix;
};

template<unsigned int Dimension>
const double ITKppDofFileReader<Dimension>::m_dPI = 3.14159265358979323846264338327950288;

#include "ITKppDofFileReader.cxx"

#endif /* ITKPPDOFFILEREADER_H_ */
