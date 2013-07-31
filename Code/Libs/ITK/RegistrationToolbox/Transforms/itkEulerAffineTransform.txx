/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkEulerAffineTransform_txx
#define __itkEulerAffineTransform_txx

#include "itkEulerAffineTransform.h"
#include <vnl/algo/vnl_cholesky.h>
#include <vnl/algo/vnl_determinant.h>
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_math.h>
#include <niftkConversionUtils.h>
#include <iostream>
#include <fstream>
#include <boost/format.hpp>
#include <itkLogHelper.h>

namespace itk
{
// Constructor with default arguments
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
EulerAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::EulerAffineTransform()
  : Superclass(OutputSpaceDimension, ParametersDimension)
{
  m_ChangeOrigin.SetIdentity(); 
  m_UnChangeOrigin.SetIdentity(); 
}


// Constructor with default arguments
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
EulerAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::EulerAffineTransform( unsigned int outputDims, 
                             unsigned int paramDims   )
  : Superclass(outputDims, paramDims)
{
  m_ChangeOrigin.SetIdentity(); 
  m_UnChangeOrigin.SetIdentity(); 
}


// Destructor
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
EulerAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::~EulerAffineTransform()
{
  return;
}

template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
void
EulerAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::ComputeComponentMatrices( void ) const
{
  double toRadians = vnl_math::pi/180.0;
  
  if ( InputSpaceDimension == 2 )
    {
      double cx=this->GetCenter()[0];
      double cy=this->GetCenter()[1];
      double rz=this->GetRotation()[0];
      double tx=this->GetTranslation()[0];
      double ty=this->GetTranslation()[1];
      double sx=this->GetScale()[0];
      double sy=this->GetScale()[1];
      double k0=this->GetSkew()[0];
      double srz=vcl_sin(rz*toRadians);
      double crz=vcl_cos(rz*toRadians);
      
      m_ChangeOrigin.SetIdentity();
      m_ChangeOrigin[0][2]=-cx;
      m_ChangeOrigin[1][2]=-cy;
      
      m_Rz.SetIdentity();
      m_Rz[0][0]=crz;
      m_Rz[0][1]=srz;
      m_Rz[1][0]=-srz;
      m_Rz[1][1]=crz;

      m_Trans.SetIdentity();
      m_Trans[0][2]=tx;
      m_Trans[1][2]=ty;

      m_Scale.SetIdentity();
      m_Scale[0][0]=sx;
      m_Scale[1][1]=sy;

      m_Skew.SetIdentity();
      m_Skew[0][1]=k0;

      m_UnChangeOrigin.SetIdentity();
      m_UnChangeOrigin[0][2]=cx;
      m_UnChangeOrigin[1][2]=cy;
      
    }
  else if ( InputSpaceDimension == 3 )
    {
      double cx=this->GetCenter()[0];
      double cy=this->GetCenter()[1];
      double cz=this->GetCenter()[2];
      double rx=this->GetRotation()[0];
      double ry=this->GetRotation()[1];
      double rz=this->GetRotation()[2];
      double tx=this->GetTranslation()[0];
      double ty=this->GetTranslation()[1];
      double tz=this->GetTranslation()[2];
      double sx=this->GetScale()[0];
      double sy=this->GetScale()[1];
      double sz=this->GetScale()[2];
      double k0=this->GetSkew()[0];
      double k1=this->GetSkew()[1];
      double k2=this->GetSkew()[2];
      double srx=vcl_sin(rx*toRadians);
      double crx=vcl_cos(rx*toRadians);
      double sry=vcl_sin(ry*toRadians);
      double cry=vcl_cos(ry*toRadians);
      double srz=vcl_sin(rz*toRadians);
      double crz=vcl_cos(rz*toRadians);
      
      m_ChangeOrigin.SetIdentity();
      m_ChangeOrigin[0][3]=-cx;
      m_ChangeOrigin[1][3]=-cy;
      m_ChangeOrigin[2][3]=-cz;
      
      m_Rz.SetIdentity();
      m_Rz[0][0]=crz;
      m_Rz[0][1]=srz;
      m_Rz[1][0]=-srz;
      m_Rz[1][1]=crz;

      m_Ry.SetIdentity();
      m_Ry[0][0]=cry;
      m_Ry[0][2]=sry;
      m_Ry[2][0]=-sry;
      m_Ry[2][2]=cry;

      m_Rx.SetIdentity();
      m_Rx[1][1]=crx;
      m_Rx[1][2]=srx;
      m_Rx[2][1]=-srx;
      m_Rx[2][2]=crx;

      m_Trans.SetIdentity();
      m_Trans[0][3]=tx;
      m_Trans[1][3]=ty;
      m_Trans[2][3]=tz;

      m_Scale.SetIdentity();
      m_Scale[0][0]=sx;
      m_Scale[1][1]=sy;
      m_Scale[2][2]=sz;
      
      m_Skew.SetIdentity();
      m_Skew[0][1]=k0;
      m_Skew[0][2]=k1;
      m_Skew[1][2]=k2;

      m_UnChangeOrigin.SetIdentity();
      m_UnChangeOrigin[0][3]=cx;
      m_UnChangeOrigin[1][3]=cy;
      m_UnChangeOrigin[2][3]=cz;
      
    }
  else
    {
      itkExceptionMacro( << "EulerAffineTransform, number of Input Dimensions, should be 2 or 3");
    }
}
 
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
const typename EulerAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>::JacobianType 
EulerAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::GetJacobian( const InputPointType &p ) const
{
  JacobianType tmp;
  this->ComputeJacobianWithRespectToParameters( p, tmp );
  return tmp;	
}
 
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
void
EulerAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::ComputeJacobianWithRespectToParameters( const InputPointType &p, JacobianType &jacobian ) const
{
  unsigned int dof = this->GetNumberOfDOF();
  double toRadians = vnl_math::pi/180.0;

  int i = 0;
  
  if ( InputSpaceDimension == 2 )
    {
      jacobian.SetSize(2, dof);
      
      double cx=this->GetCenter()[0];
      double cy=this->GetCenter()[1];
      double th=this->GetRotation()[0];
      double sx=this->GetScale()[0];
      double sy=this->GetScale()[1];
      double k0=this->GetSkew()[0];
      double sth=vcl_sin(th*toRadians);
      double cth=vcl_cos(th*toRadians);
      double one=sx*((p[0]-cx)-k0*(p[1]-cy));
      double two=sy*((p[1]-cy));
      
      if (this->GetOptimiseRotation())
        {
          // dx/dtheta, dy/dtheta
          jacobian[0][i]= -sth*one + cth*two;
          jacobian[1][i]= -cth*one - sth*two;
          i++;
        }
        
      if (this->GetOptimiseTranslation())
        {
          // dx/dtx, dy/dtx
          jacobian[0][i]=1;
          jacobian[1][i]=0;
          i++;
          
          // dy/dtx, dy/dty
          jacobian[0][i]=0;
          jacobian[1][i]=1;
          i++;          
        }
        
      if (this->GetOptimiseScale())
        {
          // dx/dsx, dy/dsx
          double oneDerivSx = (p[0]-cx) - k0*(p[1]-cy);
          jacobian[0][i]=  cth*(oneDerivSx);
          jacobian[1][i]= -sth*(oneDerivSx);
          i++;

          // dx/dsy, dy/dsy
          double twoDerivSy = (p[1]-cy);
          jacobian[0][i] = sth*(twoDerivSy);
          jacobian[1][i] = cth*(twoDerivSy);
          i++;
        }
      
      if (this->GetOptimiseSkew())
        {
          // dx/dk1, dy/dk1
          double sxTimesDerivSy = -sx*(p[1]-cy);
          jacobian[0][i] = cth*(sxTimesDerivSy);
          jacobian[1][i] = -sth*(sxTimesDerivSy);
          i++;        
        }
    }
  else if ( InputSpaceDimension == 3 )
    {
      jacobian.SetSize(3, dof);

      double cx=this->GetCenter()[0];
      double cy=this->GetCenter()[1];
      double cz=this->GetCenter()[2];
      double rx=this->GetRotation()[0];
      double ry=this->GetRotation()[1];
      double rz=this->GetRotation()[2];
      double sx=this->GetScale()[0];
      double sy=this->GetScale()[1];
      double sz=this->GetScale()[2];
      double k1=this->GetSkew()[0];
      double k2=this->GetSkew()[1];
      double k3=this->GetSkew()[2];
      double srx=vcl_sin(rx*toRadians);
      double crx=vcl_cos(rx*toRadians);
      double sry=vcl_sin(ry*toRadians);
      double cry=vcl_cos(ry*toRadians);
      double srz=vcl_sin(rz*toRadians);
      double crz=vcl_cos(rz*toRadians);
      double one=sx*((p[0]-cx) + k1*(p[1]-cy) + k2*(p[2]-cz));
      double two=sy*((p[1]-cy) + k3*(p[2]-cz));
      double three=sz*(p[2]-cz);
      
      double r1,r2,r3,r4,r5,r6,r7,r8,r9;
      r1 =  cry*crz;                r2 =  cry*srz;               r3 = -sry;
      r4 = -crx*srz + srx*sry*crz;  r5 =  crx*crz + srx*sry*srz; r6 =  srx*cry;
      r7 =  srx*srz + crx*sry*crz;  r8 = -srx*crz + crx*sry*srz; r9 =  crx*cry;
       
      if (this->GetOptimiseRotation())
        {
          
          // dx,y,z w.r.t. rx.
          jacobian[0][i] = 0;
          jacobian[1][i] = one * ( srx*srz + crx*sry*crz) + two * (crx*sry*srz - srx*crz) + three * (crx*cry);
          jacobian[2][i] = one * ( crx*srz - srx*sry*crz) + two * (-crx*crz - srx*sry*srz) + three * (-srx*cry);
          i++;

          // dx,y,z w.r.t. ry
          jacobian[0][i] = one * ( -sry*crz) + two * (-sry*srz) + three * (-cry);
          jacobian[1][i] = one * (srx*cry*crz) + two * (srx*cry*srz) + three * (-srx*sry);
          jacobian[2][i] = one * (crx*cry*crz) + two * (crx*cry*srz) + three * (-crx*sry);
          i++;

          // dx,y,z w.r.t. rz
          jacobian[0][i] = one * (-cry*srz) + two * (cry*crz);
          jacobian[1][i] = one * (-crx*crz - srx*sry*srz) + two * (-crx*srz + srx*sry*crz);
          jacobian[2][i] = one * ( srx*crz - crx*sry*srz) + two * (srx*srz + crx*sry*crz);
          i++;
           
        }
              
      if (this->GetOptimiseTranslation())
        {

          // dx,y,z w.r.t. tx
          jacobian[0][i] = 1;
          jacobian[1][i] = 0;
          jacobian[2][i] = 0;
          i++;

          // dx,y,z w.r.t. ty
          jacobian[0][i] = 0;
          jacobian[1][i] = 1;
          jacobian[2][i] = 0;
          i++;

          // dx,y,z w.r.t. tz
          jacobian[0][i] = 0;
          jacobian[1][i] = 0;
          jacobian[2][i] = 1;
          i++;
        }

      if (this->GetOptimiseScale())
        {
          // dx,y,z w.r.t. sx
          double derivOneWrtSx = ((p[0]-cx) + k1*(p[1]-cy) + k2*(p[2]-cz));
          jacobian[0][i] = r1 * derivOneWrtSx;
          jacobian[1][i] = r4 * derivOneWrtSx;
          jacobian[2][i] = r7 * derivOneWrtSx;
          i++;

          // dx,y,z w.r.t. sy
          double derivTwoWrtSy = ((p[1]-cy) + k3*(p[2]-cz));
          jacobian[0][i] = r2 * derivTwoWrtSy;
          jacobian[1][i] = r5 * derivTwoWrtSy;
          jacobian[2][i] = r8 * derivTwoWrtSy;
          i++;

          // dx,y,z w.r.t. sz
          double derivThreeWrtSz = (p[2]-cz);
          jacobian[0][i] = r3 * derivThreeWrtSz;
          jacobian[1][i] = r6 * derivThreeWrtSz;
          jacobian[2][i] = r9 * derivThreeWrtSz;
          i++;
        }

        
      if (this->GetOptimiseSkew())
        {
          // dx,y,z w.r.t. k1
          double derivOneWrtK1 = sx*(p[1]-cy);
          jacobian[0][i] = r1 * derivOneWrtK1;
          jacobian[1][i] = r4 * derivOneWrtK1;
          jacobian[2][i] = r7 * derivOneWrtK1;
          i++;

          // dx,y,z w.r.t. k2
          double derivOneWrtK2 = sx*(p[2]-cz);
          jacobian[0][i] = r1 * derivOneWrtK2;
          jacobian[1][i] = r4 * derivOneWrtK2;
          jacobian[2][i] = r7 * derivOneWrtK2;
          i++;

          // dx,y,z w.r.t. k3
          double derivTwoWrtK3 = sy*(p[2]-cz);
          jacobian[0][i] = r2 * derivTwoWrtK3;
          jacobian[1][i] = r5 * derivTwoWrtK3;
          jacobian[2][i] = r8 * derivTwoWrtK3;
          i++;
        }
    }
  else
    {
      itkExceptionMacro( << "EulerAffineTransform, number of Input Dimensions, should be 2 or 3");
    }
}

template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
void
EulerAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::ComputeMatrixAndOffset( void ) 
{

  this->ComputeComponentMatrices();
  Matrix<TScalarType,NInputDimensions+1,NInputDimensions+1> Result;	

  if (InputSpaceDimension == 3)
    {
      Result = (m_UnChangeOrigin * (m_Trans * (m_Rx * (m_Ry * (m_Rz * (m_Scale * (m_Skew * m_ChangeOrigin)))))));
    }
  else if (InputSpaceDimension == 2)
    {        
      Result =  (m_UnChangeOrigin * (m_Trans * (m_Rz * (m_Scale * (m_Skew * m_ChangeOrigin)))));
    }
  else
    {
      itkExceptionMacro( << "EulerAffineTransform, number of Input Dimensions, should be 2 or 3");
    }

  // Stick the result in m_Matrix and m_Offset;
  for (int i = 0; i < InputSpaceDimension; i++)
    {
      for (int j = 0; j < InputSpaceDimension; j++)
        {
          this->m_Matrix[i][j] = Result[i][j];
        }
      this->m_Offset[i] = Result[i][InputSpaceDimension];
    }
}

template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
void
EulerAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::ComputeParametersFromMatrixAndOffset( void ) 
{
  if (InputSpaceDimension == 3)
    {
      itkExceptionMacro( << "ComputeParametersFromMatrixAndOffset() not implemented yet.");  
    }
  else if (InputSpaceDimension == 2)
    {
      itkExceptionMacro( << "ComputeParametersFromMatrixAndOffset() not implemented yet.");  
    }
  else
    {
      itkExceptionMacro( << "EulerAffineTransform, number of Input Dimensions, should be 2 or 3");  
    }
}

template<class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
bool
EulerAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::GetInv(UCLBaseTransform< TScalarType, NInputDimensions, NOutputDimensions >* inverse) const
{
  Superclass* switchableAffineTransformInverse = dynamic_cast<Superclass*>(inverse);
  
  return this->GetInverse(switchableAffineTransformInverse);  
}

template<class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
EulerAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::InverseTransformPoint(const InputPointType & point, InputPointType& out)
{
  typename Self::Pointer inverse = Self::New();
  GetInv(inverse);
  out = inverse->TransformPoint(point);       
}
template<class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
EulerAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::SetParametersFromTransform( const FullAffineTransformType* fullAffine )
{

  int numberOfDOF = 12;
  this->SetNumberOfDOF(numberOfDOF);
  
  ParametersType P;
  P.SetSize(numberOfDOF);
  
  typedef vnl_matrix<double> VnlAffineMatrixType;
  typedef vnl_matrix<double> VnlRotationMatrixType;

  // Taken from spm_imatrix.m
  VnlAffineMatrixType matrix(NOutputDimensions+1, NInputDimensions+1);
  VnlRotationMatrixType R(NOutputDimensions, NInputDimensions);
  VnlRotationMatrixType RTranspose;
  VnlRotationMatrixType RTransposeTimesR;
  VnlRotationMatrixType C;
  
  for (unsigned int i = 0; i < NOutputDimensions+1; i++)
  {
    for (unsigned int j = 0; j < NInputDimensions+1; j++)
    {
      int index = i*(NOutputDimensions+1) + j;
      matrix[i][j] = fullAffine->GetParameters()[index];
    }
  }
  
  // We are doing the rotations around the centre of the image. 
  // We need to move the matrix to the centre of the image before decomposing the matrix. 
  // See this: 
  // Result = (m_UnChangeOrigin * (m_Trans * (m_Rx * (m_Ry * (m_Rz * (m_Scale * (m_Skew * m_ChangeOrigin)))))));
  // from ComputeMatrixAndOffset(). 
  vnl_matrix<double> unChangeOriginInverse = m_UnChangeOrigin.GetInverse(); 
  matrix = matrix*m_ChangeOrigin.GetInverse(); 
  matrix = unChangeOriginInverse*matrix; 

  for (unsigned int i = 0; i < NOutputDimensions; i++)
  {
    for (unsigned int j = 0; j < NInputDimensions; j++)
    {
      R[i][j] = matrix[i][j];
    }
  }    
  RTranspose = R.transpose();
  RTransposeTimesR = RTranspose*R;
  vnl_cholesky cholesky(RTransposeTimesR);
  C = cholesky.upper_triangle();
  
  // Set translations
  P[0] = matrix[0][3];
  P[1] = matrix[1][3];
  P[2] = matrix[2][3];

  niftkitkDebugMacro(<< "SetParameters():Decomposed translations tx=" << P[0] \
      << ", ty=" << P[1] \
      << ", tz=" << P[2] \
      );
  
  // Set scale = diagonal of C.
  P[6] = C[0][0];
  P[7] = C[1][1];
  P[8] = C[2][2];

  niftkitkDebugMacro(<< "SetParameters():Decomposed scale sx=" << P[6] \
      << ", sy=" << P[7] \
      << ", sz=" << P[8] \
      );

  if (vnl_determinant(R) < 0)
    {
      P[6] = -C[0][0];
      
      niftkitkDebugMacro(<< "SetParameters():Decomposed scale sx=" << P[6]);
    }
  
  // Work out skews
  VnlRotationMatrixType CDiag(NOutputDimensions, NInputDimensions);
  CDiag.set_identity();
  CDiag[0][0] = C[0][0];
  CDiag[1][1] = C[1][1];
  CDiag[2][2] = C[2][2];
  
  VnlRotationMatrixType CDiagInverse = vnl_matrix_inverse<double>(CDiag);
  C = CDiagInverse*C;
  P[9]  = C[0][1];
  P[10] = C[0][2];
  P[11] = C[1][2];

  niftkitkDebugMacro(<< "SetParameters():Decomposed skew k1=" << P[9] \
      << ", k2=" << P[10] \
      << ", k3=" << P[11] \
      );

  // Work out rotations
  Pointer R0Transform = Self::New();
  R0Transform->SetNumberOfDOF(numberOfDOF);
  
  ParametersType Q;
  Q.SetSize(numberOfDOF);
  Q.Fill(0);
  Q[6] = P[6];
  Q[7] = P[7];
  Q[8] = P[8];
  Q[9] = P[9];
  Q[10] = P[10];
  Q[11] = P[11];
  R0Transform->SetParameters(Q);

  for (unsigned int i = 0; i < NOutputDimensions+1; i++)
  {
    for (unsigned int j = 0; j < NInputDimensions+1; j++)
    {
      int index = i*(NOutputDimensions+1) + j;
      matrix[i][j] = R0Transform->GetFullAffineTransform()->GetParameters()[index];      
    }
  }

  VnlRotationMatrixType R0(NOutputDimensions, NInputDimensions);
  
  for (unsigned int i = 0; i < NOutputDimensions; i++)
  {
    for (unsigned int j = 0; j < NInputDimensions; j++)
    {
      R0[i][j] = matrix[i][j];
    }
  }    
  
  VnlRotationMatrixType R0Inverse = vnl_matrix_inverse<double>(R0);
  VnlRotationMatrixType R1 = R * R0Inverse;

  P[4] = vcl_asin(niftk::fixRangeTo1(R1[0][2]));
  
  double tmp;
  
  tmp = fabs(P[11]) - vnl_math::pi/2.0;
  if (tmp*tmp < 0.000000001)
    {
      P[3] = 0;
      P[5] = atan2(-1 * niftk::fixRangeTo1(R1[1][0]) , niftk::fixRangeTo1(R1[2][0] / R1[0][2]));
    }
  else
    { 
      tmp = vcl_cos(P[4]);
      P[3] = atan2(niftk::fixRangeTo1(R1[1][2]/tmp), niftk::fixRangeTo1(R1[2][2]/tmp));
      P[5] = atan2(niftk::fixRangeTo1(R1[0][1]/tmp), niftk::fixRangeTo1(R1[0][0]/tmp));
    }

  double degreesToRadians = vnl_math::pi/180.0;
  P[3] /= degreesToRadians;
  P[4] /= degreesToRadians;
  P[5] /= degreesToRadians;
  
  niftkitkDebugMacro(<< "SetParameters():Decomposed rotations rx=" << P[3] \
      << ", ry=" << P[4] \
      << ", rz=" << P[5] \
      );

  // Now set it on the transformation.
  this->SetParameters(P);
}


template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
bool
EulerAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::SaveFullAffineMatrix(std::string filename)
{
  // TODO: Robust error handling!
  
  typename FullAffineTransformType::Pointer affine = this->GetFullAffineTransform();
  std::ofstream outputFile;
  outputFile.open(filename.c_str());
  for (unsigned int i = 0; i < 4; i++)
    {
      outputFile << boost::format("%10.6f  %10.6f  %10.6f  %10.6f") % affine->GetParameters()[i*4 + 0] % affine->GetParameters()[i*4 + 1] % affine->GetParameters()[i*4 + 2] % affine->GetParameters()[i*4 + 3] << std::endl;    
    }
   outputFile.close();
   return true;
}

template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
bool
EulerAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::LoadFullAffineMatrix(std::string filename)
{
  // TODO: Robust error handling!
   
  unsigned int numberOfDOF = 16;
  bool result = false;
 
  double value;
  std::vector<double> list;
  
  std::ifstream inputFile;
  inputFile.open(filename.c_str(), std::ifstream::in); 
  
  while (!inputFile.eof())
    {
      inputFile >> value;
      if (!inputFile.fail())
      {
        list.push_back(value);
      }
    }
  inputFile.close();

  niftkitkDebugMacro(<< "LoadFullAffineMatrix():Loaded " << list.size() << " parameters");
  
  if (list.size() == numberOfDOF)
  {
    ParametersType p;
    p.SetSize(numberOfDOF);
    p.Fill(0);
    
    for (unsigned int i = 0; i < numberOfDOF; i++)
    {
      p[i] = list[i];
    }
    niftkitkDebugMacro(<< "LoadFullAffineMatrix(" << filename << "):Read transform parameters:" << p );
  
    typename FullAffineTransformType::Pointer affine = this->GetFullAffineTransform();
    affine->SetParameters(p);
  
    this->SetParametersFromTransform(affine);
    result = true;
  }
  else
  {
    niftkitkErrorMacro( "LoadFullAffineMatrix(" << filename << "):Read " << list.size() << " parameters???");
    result = false;
  }
  return result;
}

} // namespace

#endif // __itkEulerAffineTransform_txx

