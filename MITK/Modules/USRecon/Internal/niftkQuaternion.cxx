/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQuaternion.h"

#include <cmath>

// Create the quaternion from a rotation around an aixs
niftkQuaternion::niftkQuaternion(double angle, const vector<double> v) : vector<double>(4)
{
	SetAngleAxis(angle, v[0], v[1], v[2]);
}


// Euler angle to quaternion
niftkQuaternion::niftkQuaternion(double A, double B, double C, int flag) : vector<double>(4)
{
	CreateFromEulerAngles(A, B, C, flag);
}


// Public member functions:
void niftkQuaternion::Create(double a, double b, double c, double d)
{
	begin()[0] = a; begin()[1] = b; begin()[2] = c; begin()[3] = d;
}


void niftkQuaternion::CreateFromEulerAngles(double A, double B, double C, int flag)
{
	niftkQuaternion q1,q2,q3,result;

	switch (flag)
	{
	case 0:		//Z_Y_Z
		q1[0] = cos(A/2);
		q1[1] = 0;
		q1[2] = 0;
		q1[3] = sin(A/2);

		q2[0] = cos(B/2);
		q2[1] = 0;
		q2[2] = sin(B/2);
		q2[3] = 0;

		q3[0] = cos(C/2);
		q3[1] = 0;
		q3[2] = 0;
		q3[3] = sin(C/2);

		break;
	case 1:		//Z_X_Z
		q1[0] = cos(A/2);
		q1[1] = 0;
		q1[2] = 0;
		q1[3] = sin(A/2);

		q2[0] = cos(B/2);
		q2[1] = sin(B/2);
		q2[2] = 0;
		q2[3] = 0;

		q3[0] = cos(C/2);
		q3[1] = 0;
		q3[2] = 0;
		q3[3] = sin(C/2);

	  break;
	}

	result = q1*q2*q3;
	
	for (int i=0; i<4; i++)
		begin()[i] = result[i] ;

	m_Angle = acos(begin()[0]);
	m_Axis[0] = begin()[1] / sin(m_Angle);
	m_Axis[1] = begin()[2] / sin(m_Angle);
	m_Axis[2] = begin()[3] / sin(m_Angle);
	
	m_Angle *= 2;
}


double niftkQuaternion::Norm() const
{
	double q0, q1, q2, q3;

	q0 = begin()[0];
	q1 = begin()[1];
	q2 = begin()[2];
	q3 = begin()[3];

	return sqrt( q0*q0 + q1*q1 +q2*q2 +q3*q3 );
}


bool niftkQuaternion::IsUnit() const
{
	if(( this->Norm() - 1.0) < 1e-10) 
		return true;

    return false;
}

// overloaded operators:

//conjugate
niftkQuaternion niftkQuaternion::Conjugate() const
{
	niftkQuaternion result;

	result[0] = begin()[0];
	result[1] = -begin()[1];
	result[2] = -begin()[2];
	result[3] = -begin()[3];
	
	return result;
}

void niftkQuaternion::SetAngleAxis(double angle, double x, double y, double z)
{
	m_Angle = angle;
	m_Axis[0] = x;
	m_Axis[1] = y;
	m_Axis[2] = z;

	begin()[0] = cos(angle/2);
	begin()[1] = sin(angle/2) * x;
	begin()[2] = sin(angle/2) * y;
	begin()[3] = sin(angle/2) * z;
}

// add operator +
niftkQuaternion niftkQuaternion::operator + (const niftkQuaternion& q) const
{
	niftkQuaternion result;
	
	result[0] = (*this)[0] + q[0];
	result[1] = (*this)[1] + q[1];
	result[2] = (*this)[2] + q[2];
	result[3] = (*this)[3] + q[3];
	
	return result;
}


// subtraction operator -
niftkQuaternion niftkQuaternion::operator - (const niftkQuaternion& q) const
{
	niftkQuaternion result;
	
	result[0] = (*this)[0] - q[0];
	result[1] = (*this)[1] - q[1];
	result[2] = (*this)[2] - q[2];
	result[3] = (*this)[3] - q[3];
	
	return result;
}


// multiplication operator *
niftkQuaternion niftkQuaternion::operator * (const niftkQuaternion& q) const
{
	niftkQuaternion result = *this ;
	double s1,l1,m1,n1,s2,l2,m2,n2;

	s1 = (*this)[0];
	l1 = (*this)[1];
	m1 = (*this)[2];
	n1 = (*this)[3];

	s2 = q[0];
	l2 = q[1];
	m2 = q[2];
	n2 = q[3];

	result[0] = s1 * s2 - l1 * l2 - m1 * m2 - n1 * n2;
	result[1] = s1 * l2 + s2 * l1 + m1 * n2 - n1 * m2;
	result[2] = s1 * m2 + s2 * m1 + n1 * l2 - l1 * n2;
	result[3] = s1 * n2 + s2 * n1 + l1 * m2 - m1 * l2;

	return result;
}


// multiplication operator *
niftkQuaternion& niftkQuaternion::operator *= (const niftkQuaternion& q)
{
	double s1,l1,m1,n1,s2,l2,m2,n2;

	s1 = (*this)[0];
	l1 = (*this)[1];
	m1 = (*this)[2];
	n1 = (*this)[3];

	s2 = q[0];
	l2 = q[1];
	m2 = q[2];
	n2 = q[3];

	(*this)[0] = s1 * s2 - l1 * l2 - m1 * m2 - n1 * n2;
	(*this)[1] = s1 * l2 + s2 * l1 + m1 * n2 - n1 * m2;
	(*this)[2] = s1 * m2 + s2 * m1 + n1 * l2 - l1 * n2;
	(*this)[3] = s1 * n2 + s2 * n1 + l1 * m2 - m1 * l2;

	return *this;
}


void niftkQuaternion::GetAngleAxis(double& angle, double* axis)
{
	angle = acos(begin()[0]);

	if(angle!=0)
	{
		double sin_angle = sin(angle);
		axis[0] = (*this)[1] / sin_angle;
		axis[1] = (*this)[2] / sin_angle;
		axis[2] = (*this)[3] / sin_angle;
	}
	else
		axis[0] = axis[1] = axis[2] =1.0;

	angle *= 2;
}
