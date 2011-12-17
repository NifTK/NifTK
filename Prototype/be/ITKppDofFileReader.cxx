/*
 * ITKppDofFileReader.cpp
 *
 *  Created on: 4 Feb 2011
 *      Author: b-eiben
 */


#ifndef ITKPPDOFFILEREADER_CPP_
#define ITKPPDOFFILEREADER_CPP_

#include "ITKppDofFileReader.h"

/*
 * Constructor
 */
template<unsigned int Dimension>
ITKppDofFileReader<Dimension>::ITKppDofFileReader( std::string dofFileName )
  : m_homogenousMatrix()
{
#ifndef WORDS_BIGENDIAN
	m_bSwapped = true;
#else
	m_bSwapped = false;
#endif

	/*
	 * read the input file
	 */
	int   iLength;

	std::ifstream ifIn;
	ifIn.open ( dofFileName.c_str(), std::ios::binary );
	ifIn.seekg( 0, std::ios::end );
	iLength = ifIn.tellg();
	ifIn.seekg ( 0, std::ios::beg );

	char* pcCharBuf = new char[iLength];

	ifIn.read( (char*) pcCharBuf, iLength );
	ifIn.close();


	/*
	 * fill the data into the array
	 */

	m_dofsHeader = new DofFileHeader;

	memcpy( & m_dofsHeader->uiMagicNumber,        & pcCharBuf[ 0 * sizeof(unsigned int) + 0 * sizeof(double) ], sizeof( unsigned int ) );
	memcpy( & m_dofsHeader->uiTransformationType, & pcCharBuf[ 1 * sizeof(unsigned int) + 0 * sizeof(double) ], sizeof( unsigned int ) );
	memcpy( & m_dofsHeader->uiNumberOfDofs,       & pcCharBuf[ 2 * sizeof(unsigned int) + 0 * sizeof(double) ], sizeof( unsigned int ) );

	/* Correct for swapped data in file */
	if ( m_bSwapped )  this->swapFileHeaderBytes();

	m_pdDofs = new double[ m_dofsHeader->uiNumberOfDofs ];

	for ( unsigned int uiI = 0;  uiI < m_dofsHeader->uiNumberOfDofs;  ++uiI )
	{
		memcpy( & m_pdDofs[uiI], & pcCharBuf[ 3 * sizeof(unsigned int) + uiI * sizeof(double) ], sizeof( double ) );
	}

	if ( m_bSwapped ) this->swapDofBytes();

	this->CalculateHomogenousMatrix();

	/* clean up */
	if ( pcCharBuf ) delete pcCharBuf;
}




/*
 * Destructor
 */
template<unsigned int Dimension>
ITKppDofFileReader<Dimension>::~ITKppDofFileReader()
{
	if ( m_dofsHeader ) delete m_dofsHeader;
	if ( m_pdDofs     ) delete m_pdDofs;
}




/*
 * Swaps the bytes of the dof header
 */
template <unsigned int Dimension>
void ITKppDofFileReader<Dimension>::swapFileHeaderBytes()
{
	std::cout << "Swapping data..." << std::endl;

	this->swap32( (char*) ( & m_dofsHeader->uiMagicNumber        ) );
	this->swap32( (char*) ( & m_dofsHeader->uiTransformationType ) );
	this->swap32( (char*) ( & m_dofsHeader->uiNumberOfDofs         ) );

	return;
}




/*
 * Swaps the bytes of the dofs
 */
template <unsigned int Dimension>
void ITKppDofFileReader<Dimension>::swapDofBytes()
{
	for (unsigned int uiI = 0; uiI < m_dofsHeader->uiNumberOfDofs;  ++uiI )
	{
		this->swap64( (char*) & m_pdDofs[ uiI] );
	}

	return;
}






/*
 * Byte swapping of 32bit word.
 */
template <unsigned int Dimension>
void ITKppDofFileReader<Dimension>::swap32(char* pcIn)
{
	char b;

	b       = pcIn[0];
	pcIn[0] = pcIn[3];
	pcIn[3] = b;
	b       = pcIn[1];
	pcIn[1] = pcIn[2];
	pcIn[2] = b;

	return;
}




/*
 * Byte swapping of 64bit word.
 */
template <unsigned int Dimension>
void ITKppDofFileReader<Dimension>::swap64(char* pcIn)
{
	char b;

	b       = pcIn[0];
	pcIn[0] = pcIn[7];
	pcIn[7] = b;

	b       = pcIn[1];
	pcIn[1] = pcIn[6];
	pcIn[6] = b;

	b       = pcIn[2];
	pcIn[2] = pcIn[5];
	pcIn[5] = b;

	b       = pcIn[3];
	pcIn[3] = pcIn[4];
	pcIn[4] = b;

	return;
}




/*
 * Print the data
 */
template <unsigned int Dimension>
void ITKppDofFileReader<Dimension>::PrintDofs()
{
	std::cout << "dof-file contents:      "                                       << std::endl
	          << "  - MagicNumber:        " << m_dofsHeader->uiMagicNumber        << std::endl
			  << "  - TransformationType: " << m_dofsHeader->uiTransformationType << std::endl
			  << "  - NoOfDofs:           " << m_dofsHeader->uiNumberOfDofs       << std::endl;

	for (unsigned int uiI = 0;  uiI < m_dofsHeader->uiNumberOfDofs;  ++uiI )
	{
			  std::cout << "  - dof_"<< uiI  << ":              " << m_pdDofs[uiI] << std::endl;
	}
}




/*
 * Print the homogenous matrix...
 */
template <unsigned int Dimension>
void ITKppDofFileReader<Dimension>::PrintHomogenousMatrix()
{
	std::cout << m_homogenousMatrix << std::endl;

	return;
}




/*
 * Calculate the homogeneous matrix
 */
template <unsigned int Dimension>
void ITKppDofFileReader<Dimension>::CalculateHomogenousMatrix()
{
	m_homogenousMatrix.SetIdentity();

	if ( this->m_dofsHeader->uiNumberOfDofs >= 6 )
	{
		/* Precalculate some values */
		double tX   = m_pdDofs[0];
		double tY   = m_pdDofs[1];
		double tZ   = m_pdDofs[2];
		double sinX = sin( m_pdDofs[3] * m_dPI / 180.0 );
		double cosX = cos( m_pdDofs[3] * m_dPI / 180.0 );
		double sinY = sin( m_pdDofs[4] * m_dPI / 180.0 );
		double cosY = cos( m_pdDofs[4] * m_dPI / 180.0 );
		double sinZ = sin( m_pdDofs[5] * m_dPI / 180.0 );
		double cosZ = cos( m_pdDofs[5] * m_dPI / 180.0 );

		/* Fill the matrix */
		m_homogenousMatrix(0,0) = cosY * cosZ ; // cos(y)*cos(z)
		m_homogenousMatrix(0,1) = cosY * sinZ;  // cos(y)*sin(z)
		m_homogenousMatrix(0,2) = -sinY;        // -sin(y)
		m_homogenousMatrix(0,3) = tX;           // t_x

		m_homogenousMatrix(1,0) = sinX * sinY * cosZ - cosX * sinZ; // sin(x)*sin(y)*cos(z)-cos(x)*sin(z)
		m_homogenousMatrix(1,1) = sinX * sinY * sinZ + cosX * cosZ; // sin(x)*sin(y)*sin(z)+cos(x)*cos(z)
		m_homogenousMatrix(1,2) = sinX * cosY;                      // sin(x)*cos(y)
		m_homogenousMatrix(1,3) = tY;                               // t_y

		m_homogenousMatrix(2,0) = sinX * sinZ + cosX * sinY * cosZ; // sin(x)*sin(z)+cos(x)*sin(y)*cos(z)
		m_homogenousMatrix(2,1) = cosX * sinY * sinZ - sinX * cosZ; // cos(x)*sin(y)*sin(z)-sin(x)*cos(z)
		m_homogenousMatrix(2,2) = cosX * cosY;                      // cos(x)*cos(y)
		m_homogenousMatrix(2,3) = tZ;                               // t_z
	}
	if ( this->m_dofsHeader->uiNumberOfDofs >= 12 )
	{
		double sX = m_pdDofs[6];
		double sY = m_pdDofs[7];
		double sZ = m_pdDofs[8];

		double sXY = m_pdDofs[ 9];
		double sYZ = m_pdDofs[10];
		double sXZ = m_pdDofs[11];

		
		// Skewing
		// - generate matrix
		HomogenousMatrixType skewMatrix;
		skewMatrix.SetIdentity();

		skewMatrix(0, 1) = tan( sXY * ( m_dPI / 180.0 ) );
		skewMatrix(0, 2) = tan( sXZ * ( m_dPI / 180.0 ) );
		skewMatrix(1, 2) = tan( sYZ * ( m_dPI / 180.0 ) );

		// - combine skewnig with rigid transform 
		m_homogenousMatrix = m_homogenousMatrix * skewMatrix;
		
		// Scaling
		// - generate matrix
		HomogenousMatrixType scaleMatrix;
		scaleMatrix.SetIdentity();
		scaleMatrix(0, 0) = sX / 100.;
		scaleMatrix(1, 1) = sY / 100.;
		scaleMatrix(2, 2) = sZ / 100.;

		m_homogenousMatrix = m_homogenousMatrix * scaleMatrix;
	}
	return;
}




/*
 * Returns the homogenous matrix from the dof-parameters
 */
template <unsigned int Dimension>
typename ITKppDofFileReader<Dimension>::HomogenousMatrixType ITKppDofFileReader<Dimension>::GetHomogenousMatrix()
{
	return m_homogenousMatrix;
}




#endif // ITKPPDOFFILEREADER_H_
