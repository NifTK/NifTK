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
#ifndef __itkBSplineTransform_txx
#define __itkBSplineTransform_txx

#include "itkNumericTraits.h"
#include "itkMatrixOffsetTransformBase.h"
#include "itkIdentityTransform.h"
#include "vnl/algo/vnl_matrix_inverse.h"
#include "ConversionUtils.h"
#include <iostream>

#include "itkLogHelper.h"

namespace itk
{
// Constructor with default arguments
template<class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::BSplineTransform()
{
  this->m_Grid = GridImageType::New();
  this->m_OldGrid = GridImageType::New();
  this->m_BendingEnergyGrid = BendingEnergyImageType::New();
  this->m_BendingEnergyHasBeenUpdatedFlag = false;
  this->m_BendingEnergyDerivativeFilter = BendingEnergyDerivativeFilterType::New();
  
  // Filling lookup table.
  
  m_Lookup.SetSize(s_LookupTableRows, s_LookupTableCols);
  m_Lookup1stDerivative.SetSize(s_LookupTableRows, s_LookupTableCols);
  m_Lookup2ndDerivative.SetSize(s_LookupTableRows, s_LookupTableCols);
  
  TScalarType       u = 0;
  unsigned int i = 0;
  unsigned int j = 0;
  
  for (i = 0; i < s_LookupTableRows; i++)
    {
      u = i/(TScalarType)s_LookupTableSize;
      
      for (j = 0; j < s_LookupTableCols; j++)
        {
          this->m_Lookup[i][j]              = this->B(j, 0, u);
          this->m_Lookup1stDerivative[i][j] = this->B(j, 1, u);
          this->m_Lookup2ndDerivative[i][j] = this->B(j, 2, u);          
        }
    }  
  niftkitkDebugMacro(<< "BSplineTransform():Constructed, static LUT size (" << s_LookupTableRows << "," << s_LookupTableCols << ")");
}

// Destructor
template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::~BSplineTransform()
{
  return;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::PrintSelf(std::ostream &os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);  
  os << indent << "Grid of control points: " << std::endl << m_Grid << std::endl;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
typename BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>::OutputPointType
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::TransformPoint(const InputPointType  &point ) const
{
  OutputPointType result;
  DeformationFieldIndexType index;
  DeformationFieldPixelType pixel;
  ContinuousIndex< double, NDimensions > imageDeformation;
  OutputPointType physicalDeformation; 
  DeformationFieldSpacingType spacing = this->m_DeformationField->GetSpacing();
  
  if (this->m_DeformationField->TransformPhysicalPointToIndex(point, index))
    {
      pixel = this->m_DeformationField->GetPixel(index);

      // Transform the deformation from image space to physical/world space. 
      for (unsigned int i = 0; i < NDimensions; i++)
      {
        imageDeformation[i] = pixel[i]/spacing[i];
      }
      this->m_DeformationField->TransformContinuousIndexToPhysicalPoint(imageDeformation, physicalDeformation);
      
      for (unsigned int i = 0; i < NDimensions; i++)
        {
          result[i] = point[i] + physicalDeformation[i];
        }
    }
    
  if (this->m_GlobalTransform.GetPointer() != 0)
    {
      result = this->m_GlobalTransform->TransformPoint(result);
    }
  
  return result;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::SetIdentity( void )
{
  niftkitkDebugMacro(<< "SetIdentity():Started");
  
  Superclass::SetIdentity();
  
  GridPixelType gridValue;
  gridValue.Fill(0);
  
  this->m_Grid->FillBuffer(gridValue);
  this->m_OldGrid->FillBuffer(gridValue);
  this->m_BendingEnergyGrid->FillBuffer(0);
  
  niftkitkDebugMacro(<< "SetIdentity():Finished, set m_Grid to zero, baseclass does m_Parameters");
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void 
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::InitializeGrid(FixedImagePointer fixedImage, 
    GridRegionType gridRegion,
    GridSpacingType gridSpacing,
    GridDirectionType gridDirection,
    GridOriginType gridOrigin)
{
  niftkitkDebugMacro(<< "InitializeGrid():Started");
  
  // Setup deformation field.
  Superclass::Initialize(fixedImage);

  this->m_Grid->SetRegions(gridRegion);
  this->m_Grid->SetSpacing(gridSpacing);
  this->m_Grid->SetDirection(gridDirection);
  this->m_Grid->SetOrigin(gridOrigin);
  this->m_Grid->Allocate();
  this->m_Grid->Update();

  /** m_OldGrid and m_BendingEnergyGrid will be initialized in a lazy fashion.*/
  /** So, here we just make sure they have some size, so other methods work. */
  GridRegionType dummyRegion;
  GridSizeType   dummySize;
  GridIndexType  dummyIndex;
  dummySize.Fill(1);
  dummyIndex.Fill(0);
  dummyRegion.SetSize(dummySize);
  dummyRegion.SetIndex(dummyIndex);
  
  this->m_OldGrid->SetRegions(dummyRegion);
  this->m_OldGrid->SetSpacing(gridSpacing);
  this->m_OldGrid->SetDirection(gridDirection);
  this->m_OldGrid->SetOrigin(gridOrigin);
  this->m_OldGrid->Allocate();
  this->m_OldGrid->Update();

  this->m_BendingEnergyGrid->SetRegions(dummyRegion);
  this->m_BendingEnergyGrid->SetSpacing(gridSpacing);
  this->m_BendingEnergyGrid->SetDirection(gridDirection);
  this->m_BendingEnergyGrid->SetOrigin(gridOrigin);
  this->m_BendingEnergyGrid->Allocate();
  this->m_BendingEnergyGrid->Update();

  niftkitkDebugMacro(<< "InitializeGrid():Set initial deformation grid to size=" << m_Grid->GetLargestPossibleRegion().GetSize() \
    << ", spacing=" << m_Grid->GetSpacing() \
    << ", origin=" << m_Grid->GetOrigin() \
    << ", old grid size=" << m_OldGrid->GetLargestPossibleRegion().GetSize() \
    << ", spacing=" << m_OldGrid->GetSpacing() \
    << ", origin=" << m_OldGrid->GetOrigin() \
    << ", bending energy size=" << m_BendingEnergyGrid->GetLargestPossibleRegion().GetSize() \
    << ", spacing=" << m_BendingEnergyGrid->GetSpacing() \
    << ", origin=" << m_BendingEnergyGrid->GetOrigin() \
    << ", deformation field size=" << this->m_DeformationField->GetLargestPossibleRegion().GetSize() \
    << ", spacing=" << this->m_DeformationField->GetSpacing() \
    << ", origin=" << this->m_DeformationField->GetOrigin());

  // Parameters array should match grid.
  this->ResizeParametersArray(this->m_Grid);
  
  // Reset the lot.
  this->SetIdentity();
  
  niftkitkDebugMacro(<< "InitializeGrid():Finished");
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void 
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::Initialize(FixedImagePointer image, GridSpacingType gridSpacingInMillimetres, int numberOfLevels)
{
  niftkitkDebugMacro(<< "Initialize():Started, final spacing=" << gridSpacingInMillimetres \
    << " (in mm), image size=" << image->GetLargestPossibleRegion().GetSize() \
    << ", image spacing=" << image->GetSpacing()
    << ", image origin=" << image->GetOrigin());    
  
  // Read these values off the image.
  DeformationFieldSpacingType spacing = image->GetSpacing();
  DeformationFieldDirectionType direction = image->GetDirection();
  DeformationFieldOriginType origin = image->GetOrigin();
  DeformationFieldSizeType size = image->GetLargestPossibleRegion().GetSize();
  DeformationFieldIndexType index = image->GetLargestPossibleRegion().GetIndex();
  
  // Now set up control point grid.  
  GridRegionType gridRegion;
  GridSizeType gridSize;
  GridIndexType gridIndex;
  GridSpacingType gridSpacing;
  GridDirectionType gridDirection;
  GridOriginType gridOrigin;

  for (unsigned int i = 0; i < NDimensions; i++)
    {
      gridIndex[i] = 0;
      gridSize[i] = 1 + (int)(size[i]*spacing[i]/(gridSpacingInMillimetres[i] * std::pow(2.0,numberOfLevels-1)));
      gridSpacing[i] = (size[i]/((TScalarType)(gridSize[i]-1))) * spacing[i];
      gridOrigin[i] = GetNewOrigin(size[i], spacing[i], origin[i], gridSize[i], gridSpacing[i]);      
    }

  gridDirection = direction;
  gridRegion.SetSize(gridSize);
  gridRegion.SetIndex(gridIndex);

  // And now set up grid
  this->InitializeGrid(image, gridRegion, gridSpacing, gridDirection, gridOrigin);

  niftkitkDebugMacro(<< "Initialize():Finished");
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::SetParameters(const ParametersType & parameters)
{
  niftkitkDebugMacro(<< "SetParameters():Starting with:" << parameters.GetSize() << " parameters, and parameters object:" << &parameters);

  // This will copy the array into this object.
  this->m_Parameters = parameters;

  niftkitkDebugMacro(<< "SetParameters():Done copying to parameter array.");
  
  // And this copies it into the grid.
  this->MarshallParametersToImage(this->m_Grid);
  
  niftkitkDebugMacro(<< "SetParameters():Done marshalling into grid, now updating vector field");

  if (NDimensions == 2)
    {
      this->InterpolateDeformationField2D();  
    }
  else if (NDimensions == 3)
    {
      this->InterpolateDeformationField3DMarc();  
    } 
  else 
    {
      itkExceptionMacro(<<"Wrong number of dimensions, this class only supports 2D or 3D transforms");
    } 

  // Just forcing this to make sure if you ask for Jacobian, its up to date.
  this->m_DeformationField->Modified();
  this->m_Grid->Modified();
  this->Modified();
            
  niftkitkDebugMacro(<< "SetParameters():Finished, deformation field size:" << this->m_DeformationField->GetLargestPossibleRegion().GetSize() \
      << ", and grid size:" << this->m_Grid->GetLargestPossibleRegion().GetSize());        
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
BSplineTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::InterpolateDeformationField2D()
{

  // Store these locally for performance reasons.
  DeformationFieldOriginType  fieldOrigin  = this->m_DeformationField->GetOrigin();
  DeformationFieldSpacingType fieldSpacing = this->m_DeformationField->GetSpacing();
  DeformationFieldSizeType    fieldSize    = this->m_DeformationField->GetLargestPossibleRegion().GetSize();
  GridOriginType              gridOrigin   = m_Grid->GetOrigin();  
  GridSpacingType             gridSpacing  = m_Grid->GetSpacing();  
  GridSizeType                gridSize     = m_Grid->GetLargestPossibleRegion().GetSize();

  niftkitkDebugMacro(<< "InterpolateDeformationField2D():Started, field size:" << fieldSize \
      << ", fieldSpacing:" << fieldSpacing \
      << ", fieldOrigin:" << fieldOrigin \
      << ", gridSize:" << gridSize \
      << ", gridSpacing:" << gridSpacing \
      << ", gridOrigin:" << gridOrigin );

  DeformationFieldIndexType deformationFieldIndex; // We iterate through every voxel in grid.
  InputPointType            deformationFieldPoint; // This is the world coordinate, of the previous point.  
  GridIndexType             gridClosestIndex;      // This is i, j, k in Daniel's paper, before we subtract 1.
  GridIndexType             gridIndex;             // This is i, j, k in Daniel's paper.
  GridIndexType             gridMovingIndex;       // This is for as we move around the BSpline window.
  GridSpacingType           gridBasis;             // This is u, v, w in Daniel's paper.
  GridIndexType             gridRoundedBasis;      // We round u,v,w, to lookup value in a lookup table.
  GridVoxelCoordinateType   gridVoxelCoordinate;
  
  GridPixelType             controlPointDisplacementVector; 
  DeformationFieldPixelType accumulatedDisplacementVector;
  
  DeformationFieldIteratorType iterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion());
  
  unsigned int d = 0;  // iterating over dimensions;
  unsigned int l = 0;  // l in Daniel's paper
  unsigned int m = 0;  // m in Daniel's paper
  
  TScalarType Bl = 0;       // Also in Daniel's paper
  TScalarType Bm = 0;       // Also in Daniel's paper

  GridSizeType   size;
  size[0] = 4;
  size[1] = 4;

  iterator.GoToBegin();
  while(!iterator.IsAtEnd())
    {
      deformationFieldIndex = iterator.GetIndex();
      
      // This bit is slow, so I removed this:
      //this->m_DeformationField->TransformIndexToPhysicalPoint( deformationFieldIndex, deformationFieldPoint );
      //this->m_Grid->TransformPhysicalPointToContinuousIndex( deformationFieldPoint, gridVoxelCoordinate );
      // and put the variables deformationFieldPoint, gridVoxelCoordinate in the loop below.
      
      for (d = 0; d < NDimensions; d++)
        {
          deformationFieldPoint[d] = (deformationFieldIndex[d] * fieldSpacing[d]) + fieldOrigin[d];
          gridVoxelCoordinate[d]   = (deformationFieldPoint[d] - gridOrigin[d]) / gridSpacing[d];
          gridClosestIndex[d]      = (int)floor(gridVoxelCoordinate[d]);
          gridBasis[d]             = gridVoxelCoordinate[d] - gridClosestIndex[d];
          gridIndex[d]             = gridClosestIndex[d] - 1;
          gridRoundedBasis[d]      = (int)niftk::Round(gridBasis[d]*s_LookupTableSize);
          
          // Dont forget to reset the displacement vector to zero:
          accumulatedDisplacementVector[d] = 0;
        }

      GridRegionType region;
      region.SetSize(size);      
      region.SetIndex(gridIndex);

      GridConstIteratorType gridIterator(this->m_Grid, region);
      gridIterator.GoToBegin();
      
      for (l = 0; l < 4; l++)
        {
          Bl = this->m_Lookup[gridRoundedBasis[1]][l];
          
          for (m = 0; m < 4; m++)
            {
              Bm = this->m_Lookup[gridRoundedBasis[0]][m];
              
              gridMovingIndex = gridIterator.GetIndex();
              
              if (gridMovingIndex[0] >= (int)0
               && gridMovingIndex[1] >= (int)0
               && gridMovingIndex[0] < (int) gridSize[0]
               && gridMovingIndex[1] < (int) gridSize[1])
                {
                  controlPointDisplacementVector = gridIterator.Get();
                  
                  accumulatedDisplacementVector[0] += (Bl * Bm * controlPointDisplacementVector[0]);
                  accumulatedDisplacementVector[1] += (Bl * Bm * controlPointDisplacementVector[1]);                  
                } // end if
              
              ++gridIterator;

            } // end for m
        } // end for l
        
      // Store resultant vector
      iterator.Set(accumulatedDisplacementVector);
      
      // Dont forget this!
      ++iterator;
      
    } // end while 

  niftkitkDebugMacro(<< "InterpolateDeformationField2D():Finished, field size:" << fieldSize << ", gridSize:" << gridSize);
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
BSplineTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::InterpolateDeformationField3DMarc()
{
  // Store these locally for performance reasons.
  DeformationFieldOriginType  fieldOrigin  = this->m_DeformationField->GetOrigin();
  DeformationFieldSpacingType fieldSpacing = this->m_DeformationField->GetSpacing();
  DeformationFieldSizeType    fieldSize    = this->m_DeformationField->GetLargestPossibleRegion().GetSize();
  GridOriginType              gridOrigin   = m_Grid->GetOrigin();  
  GridSpacingType             gridSpacing  = m_Grid->GetSpacing(); 
  GridSizeType                gridSize     = m_Grid->GetLargestPossibleRegion().GetSize();
  GridIndexType               gridIndex;
  GridIndexType               gridMovingIndex;
  GridPixelType               controlPointDisplacementVector; 
  DeformationFieldPixelType   accumulatedDisplacementVector;
  DeformationFieldIteratorType iterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion());
  iterator.GoToBegin();
  
  GridSpacingType             gridVoxelSpacing;
  for (unsigned int i = 0; i < NDimensions; i++)
    {
      gridVoxelSpacing[i] = (float)fieldSize[i]/(float)(gridSize[i]-1);  
    }

  /*
  printf("Matt\t:gridVoxelSpacing=%f, %f, %f\n", gridVoxelSpacing[0], gridVoxelSpacing[1], gridVoxelSpacing[2]);
  */
  
  // Must use an iterator to go through neighborhood.
  GridSizeType   size;
  size[0] = 4;
  size[1] = 4;
  size[2] = 4;

  niftkitkDebugMacro(<< "InterpolateDeformationField3DMarc():Started, field size:" << fieldSize << ", gridSize:" << gridSize);

  /*
  int tempIndex=0;
  GridConstIteratorType gridIterator(this->m_Grid, this->m_Grid->GetLargestPossibleRegion());
  gridIterator.GoToBegin();

  while(!gridIterator.IsAtEnd())
    {
      controlPointDisplacementVector=gridIterator.Get();
      printf("Matt:\t%d, %f, %f, %f\n", tempIndex, controlPointDisplacementVector[0], controlPointDisplacementVector[1], controlPointDisplacementVector[2]);
      tempIndex++;
      ++gridIterator;
    }
  */
  
  float oldBasis=1.1;
#if _USE_SSE
  union u{
  __m128 m;
  float f[4];
  } val;
#endif
  
  int index=0;
  int coord=0;
  
  float zBasis[4];
  float yzBasis[16];
  float xyzBasis[64];
  float xControlPointCoordinates[64];
  float yControlPointCoordinates[64];
  float zControlPointCoordinates[64];

  for(unsigned int z=0;z<fieldSize[2];z++){
    int zPre=(int)((float)z/gridVoxelSpacing[2]);
    float basis=(float)z/gridVoxelSpacing[2]-(float)zPre;
    if(basis<0.0) basis=0.0; //rounding error
    float FF= basis*basis;
    float FFF= FF*basis;
    float MF=1.0-basis;
    zBasis[0] = (MF)*(MF)*(MF)/6.0;
    zBasis[1] = (3.0*FFF - 6.0*FF +4.0)/6.0;
    zBasis[2] = (-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0;
    zBasis[3] = FFF/6.0;

    for(unsigned int y=0;y<fieldSize[1];y++){
      int yPre=(int)((float)y/gridVoxelSpacing[1]);
      basis=(float)y/gridVoxelSpacing[1]-(float)yPre;
      if(basis<0.0) basis=0.0; //rounding error
      float FF= basis*basis;
      float FFF= FF*basis;
      float MF=1.0-basis;
#if _USE_SSE
      val.f[0] = (MF)*(MF)*(MF)/6.0;
      val.f[1] = (3.0*FFF - 6.0*FF +4.0)/6.0;
      val.f[2] = (-3.0*FFF + 3.0*FF + 3.0*basis +1.0)/6.0;
      val.f[3] = FFF/6.0;
      __m128 tempCurrent=val.m;
      __m128* ptrBasis   = (__m128 *) &yzBasis[0];
      for(unsigned int a=0;a<4;a++){
        val.m=_mm_set_ps1(zBasis[a]);
        *ptrBasis=_mm_mul_ps(tempCurrent,val.m);
        ptrBasis++;
      }
#else
      float temp[4];
      temp[0] = (MF)*(MF)*(MF)/6.0;
      temp[1] = (3.0*FFF - 6.0*FF +4.0)/6.0;
      temp[2] = (-3.0*FFF + 3.0*FF + 3.0*basis +1.0)/6.0;
      temp[3] = FFF/6.0;
      coord=0;
      for(int a=0;a<4;a++){
        yzBasis[coord++]=temp[0]*zBasis[a];
        yzBasis[coord++]=temp[1]*zBasis[a];
        yzBasis[coord++]=temp[2]*zBasis[a];
        yzBasis[coord++]=temp[3]*zBasis[a];
      }     
#endif

      for(unsigned int x=0;x<fieldSize[0];x++){
        
        int xPre=(int)((float)x/gridVoxelSpacing[0]);
        basis=(float)x/gridVoxelSpacing[0]-(float)xPre;
        if(basis<0.0) basis=0.0; //rounding error
        float FF= basis*basis;
        float FFF= FF*basis;
        float MF=1.0-basis;
#if _USE_SSE
        val.f[0] = (MF)*(MF)*(MF)/6.0;
        val.f[1] = (3.0*FFF - 6.0*FF +4.0)/6.0;
        val.f[2] = (-3.0*FFF + 3.0*FF + 3.0*basis +1.0)/6.0;
        val.f[3] = FFF/6.0;
        tempCurrent=val.m;      
        ptrBasis   = (__m128 *) &xyzBasis[0];
        for(int a=0;a<16;++a){
          val.m=_mm_set_ps1(yzBasis[a]);
          *ptrBasis=_mm_mul_ps(tempCurrent,val.m);
          ptrBasis++;
        }
#else
        temp[0] = (MF)*(MF)*(MF)/6.0;
        temp[1] = (3.0*FFF - 6.0*FF +4.0)/6.0;
        temp[2] = (-3.0*FFF + 3.0*FF + 3.0*basis +1.0)/6.0;
        temp[3] = FFF/6.0;
        coord=0;
        for(unsigned int a=0;a<16;a++){
          xyzBasis[coord++]=temp[0]*yzBasis[a];
          xyzBasis[coord++]=temp[1]*yzBasis[a];
          xyzBasis[coord++]=temp[2]*yzBasis[a];
          xyzBasis[coord++]=temp[3]*yzBasis[a];
        }
#endif
        /*
        printf("Matt\t:Marshalling cp, index=%d, basis=%f, oldBasis=%f, x=%d, xPre=%d, gridSpacing=%f\n", index, basis, oldBasis, x, xPre, gridVoxelSpacing[0]);
        */
        
        // The control point coordinates are exported when the interest control points are changing
        if(basis<=oldBasis || x==0){
          memset(xControlPointCoordinates,0,64*sizeof(float));
          memset(yControlPointCoordinates,0,64*sizeof(float));
          memset(zControlPointCoordinates,0,64*sizeof(float));
          
          gridIndex[0] = xPre-1;
          gridIndex[1] = yPre-1;
          gridIndex[2] = zPre-1;

          GridRegionType region;
          region.SetSize(size);      
          region.SetIndex(gridIndex);
          
          GridConstIteratorType gridIterator(this->m_Grid, region);
          gridIterator.GoToBegin();

          coord=0;
          while(!gridIterator.IsAtEnd())
            {
              gridMovingIndex = gridIterator.GetIndex();
              if (-1<gridMovingIndex[0] && -1<gridMovingIndex[1] && -1<gridMovingIndex[2] && gridMovingIndex[0]< (int)(gridSize[0]) && gridMovingIndex[1]< (int)(gridSize[1]) && gridMovingIndex[2]< (int)(gridSize[2]))
                {
                  controlPointDisplacementVector = gridIterator.Get();
                  xControlPointCoordinates[coord] = controlPointDisplacementVector[0];
                  yControlPointCoordinates[coord] = controlPointDisplacementVector[1];
                  zControlPointCoordinates[coord] = controlPointDisplacementVector[2];
                }
              ++gridIterator;
              /*
              printf("Matt\t:Marshalling cp, index=%d, coord=%d, %d, %d, %d, %f, %f, %f\n", index, coord, gridMovingIndex[0], gridMovingIndex[1], gridMovingIndex[2], xControlPointCoordinates[coord], yControlPointCoordinates[coord], zControlPointCoordinates[coord]);
              */              
              coord++;
            }
        }
        oldBasis=basis;

        /* *** B-Spline interpolation *** */
        float xReal=0.0;
        float yReal=0.0;
        float zReal=0.0;
#if _USE_SSE
        __m128 tempX =  _mm_set_ps1(0.0);
        __m128 tempY =  _mm_set_ps1(0.0);
        __m128 tempZ =  _mm_set_ps1(0.0);
        __m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
        __m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
        __m128 *ptrZ = (__m128 *) &zControlPointCoordinates[0];
        ptrBasis   = (__m128 *) &xyzBasis[0];
        //addition and multiplication of the 64 basis value and CP displacement for each axis
        for(int a=0; a<16; a++){
          tempX = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrX), tempX );
          tempY = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrY), tempY );
          tempZ = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrZ), tempZ );
          ptrBasis++;
          ptrX++;
          ptrY++;
          ptrZ++;
        }
        //the values stored in SSE variables are transfered to normal float
        val.m=tempX;
        xReal=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempY;
        yReal= val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempZ;
        zReal= val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
        //printf("Matt:\tindex=%d, previous=%f, %f, %f\n", index, currentDisplacementVector[0], currentDisplacementVector[1], currentDisplacementVector[2]);

        for(int a=0; a<64; a++){
          xReal += xControlPointCoordinates[a] * xyzBasis[a];
          yReal += yControlPointCoordinates[a] * xyzBasis[a];
          zReal += zControlPointCoordinates[a] * xyzBasis[a];
          //if (a==32)
          //  {
              //printf("Matt:\tindex=%d, %f, %f, %f, %f, %f, %f, %f\n", index, xControlPointCoordinates[a], yControlPointCoordinates[a], zControlPointCoordinates[a], xyzBasis[a], xReal, yReal, zReal);              
          //  }                    
        }
#endif
        accumulatedDisplacementVector[0] = xReal;
        accumulatedDisplacementVector[1] = yReal;
        accumulatedDisplacementVector[2] = zReal;
        //printf("Matt:\tindex=%d, %f, %f, %f\n", index, accumulatedDisplacementVector[0], accumulatedDisplacementVector[1], accumulatedDisplacementVector[2]);
        iterator.Set(accumulatedDisplacementVector);
        ++iterator;     
        index++;
      }
    }
  }

  niftkitkDebugMacro(<< "InterpolateDeformationField3DMarc():Finished, field size:" << fieldSize << ", gridSize:" << gridSize);
  
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
BSplineTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::InterpolateDeformationField3DDaniel()
{
  
  // Store these locally for performance reasons.
  DeformationFieldOriginType  fieldOrigin  = this->m_DeformationField->GetOrigin();
  DeformationFieldSpacingType fieldSpacing = this->m_DeformationField->GetSpacing();
  DeformationFieldSizeType    fieldSize    = this->m_DeformationField->GetLargestPossibleRegion().GetSize();
  GridOriginType              gridOrigin   = m_Grid->GetOrigin();  
  GridSpacingType             gridSpacing  = m_Grid->GetSpacing();  
  GridSizeType                gridSize     = m_Grid->GetLargestPossibleRegion().GetSize();

  niftkitkDebugMacro(<< "InterpolateDeformationField3DDaniel():Started, field size:" << fieldSize << ", gridSize:" << gridSize);

  DeformationFieldIndexType deformationFieldIndex; // We iterate through every voxel in grid.
  InputPointType            deformationFieldPoint; // This is the world coordinate, of the previous point.  
  GridIndexType             gridClosestIndex;      // This is i, j, k in Daniel's paper, before we subtract 1.
  GridIndexType             gridIndex;             // This is i, j, k in Daniel's paper.
  GridIndexType             gridMovingIndex;       // This is for as we move around the BSpline window.
  GridSpacingType           gridBasis;             // This is u, v, w in Daniel's paper.
  GridIndexType             gridRoundedBasis;      // We round u,v,w, to lookup value in a lookup table.
  GridVoxelCoordinateType   gridVoxelCoordinate;
  
  GridPixelType             controlPointDisplacementVector; 
  DeformationFieldPixelType accumulatedDisplacementVector;
      
  DeformationFieldIteratorType iterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion());
  
  unsigned int d = 0;  // iterating over dimensions;
  unsigned int l = 0;  // l in Daniel's paper
  unsigned int m = 0;  // m in Daniel's paper
  unsigned int n = 0;  // n in Daniel's paper
  
  TScalarType Bl = 0;       // Also in Daniel's paper
  TScalarType Bm = 0;       // Also in Daniel's paper
  TScalarType Bn = 0;       // Also in Daniel's paper

  // Must use an iterator to go through neighborhood.
  GridSizeType   size;
  size[0] = 4;
  size[1] = 4;
  size[2] = 4;

  iterator.GoToBegin();
  while(!iterator.IsAtEnd())
    {
      deformationFieldIndex = iterator.GetIndex();

      // This bit is slow, so I removed this:
      //this->m_DeformationField->TransformIndexToPhysicalPoint( deformationFieldIndex, deformationFieldPoint );
      //this->m_Grid->TransformPhysicalPointToContinuousIndex( deformationFieldPoint, gridVoxelCoordinate );
      // and put the variables deformationFieldPoint, gridVoxelCoordinate in the loop below.
      
      for (d = 0; d < NDimensions; d++)
        {
          deformationFieldPoint[d] = (deformationFieldIndex[d] * fieldSpacing[d]) + fieldOrigin[d];
          gridVoxelCoordinate[d]   = (deformationFieldPoint[d] - gridOrigin[d]) / gridSpacing[d];        
          gridClosestIndex[d]      = (int)floor(gridVoxelCoordinate[d]);
          gridBasis[d]             = gridVoxelCoordinate[d] - gridClosestIndex[d];
          gridIndex[d]             = gridClosestIndex[d] - 1;
          gridRoundedBasis[d]      = (int)round(gridBasis[d]*s_LookupTableSize);
          
          // Dont forget to reset the displacement vector to zero:
          accumulatedDisplacementVector[d] = 0;
        }

      GridRegionType region;
      region.SetSize(size);      
      region.SetIndex(gridIndex);
      
      GridConstIteratorType gridIterator(this->m_Grid, region);
      gridIterator.GoToBegin();

      for (l = 0; l < 4; l++)
        {
          Bl = this->m_Lookup[gridRoundedBasis[2]][l];
          
          for (m = 0; m < 4; m++)
            {
              Bm = this->m_Lookup[gridRoundedBasis[1]][m];
              
              for (n = 0; n < 4; n++)
                {

                  Bn = this->m_Lookup[gridRoundedBasis[0]][n];
                  
                  gridMovingIndex = gridIterator.GetIndex();
                  
                  if (gridMovingIndex[0] >= (int)0 && gridMovingIndex[0] < (int)gridSize[0]
                   && gridMovingIndex[1] >= (int)0 && gridMovingIndex[1] < (int)gridSize[1]
                   && gridMovingIndex[2] >= (int)0 && gridMovingIndex[2] < (int)gridSize[2])
                    {
                      controlPointDisplacementVector = gridIterator.Get();

                      accumulatedDisplacementVector[0] += (Bl * Bm * Bn * controlPointDisplacementVector[0]);
                      accumulatedDisplacementVector[1] += (Bl * Bm * Bn * controlPointDisplacementVector[1]);
                      accumulatedDisplacementVector[2] += (Bl * Bm * Bn * controlPointDisplacementVector[2]);                          
                    } // end if
                  
                  ++gridIterator;
                  
                } // end for n
            } // end for m
        } // end for l

      // Store resultant vector
      iterator.Set(accumulatedDisplacementVector);

      // Dont forget this!
      ++iterator;
      
    } // end while
    
  niftkitkDebugMacro(<< "InterpolateDeformationField3DDaniel():Finished, field size:" << fieldSize << ", gridSize:" << gridSize);
  
} // end function


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
typename BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>::MeasureType 
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::GetBendingEnergy()
{
  MeasureType bendingEnergy = 0;
  TScalarType bendingEnergyDivisor = NDimensions;
  
  GridSizeType gridSize = m_Grid->GetLargestPossibleRegion().GetSize();
  BendingEnergyImageSizeType bendingEnergySize = this->m_BendingEnergyGrid->GetLargestPossibleRegion().GetSize();
  
  if (gridSize != bendingEnergySize)
    {
      // Now we need to allocate the size of the bending energy grid.
      GridRegionType gridRegion       = this->m_Grid->GetLargestPossibleRegion();
      GridSizeType gridSize           = gridRegion.GetSize();
      GridIndexType gridIndex         = gridRegion.GetIndex();
      GridSpacingType gridSpacing     = this->m_Grid->GetSpacing();
      GridDirectionType gridDirection = this->m_Grid->GetDirection();
      GridOriginType gridOrigin       = this->m_Grid->GetOrigin();
      
      this->m_BendingEnergyGrid->SetRegions(gridRegion);
      this->m_BendingEnergyGrid->SetSpacing(gridSpacing);
      this->m_BendingEnergyGrid->SetOrigin(gridOrigin);
      this->m_BendingEnergyGrid->SetDirection(gridDirection);
 
      this->m_BendingEnergyGrid->Allocate();
      this->m_BendingEnergyGrid->Update();

      niftkitkDebugMacro(<< "GetBendingEnergy():Resized bending energy grid to size=" << m_BendingEnergyGrid->GetLargestPossibleRegion().GetSize() \
        << ", spacing=" << m_BendingEnergyGrid->GetSpacing() \
        << ", origin=" << m_BendingEnergyGrid->GetOrigin() \
        << ", direction=" << m_BendingEnergyGrid->GetDirection()
      );
    }

  // Work out total number of voxels * Dimensions.
  for (unsigned int i = 0; i < NDimensions; i++)
    {
      bendingEnergyDivisor *= ((TScalarType)gridSize[i]);
    }
  
  // Now call 2D or 3D versions.
  if (NDimensions == 2)
    {
      bendingEnergy = this->GetBendingEnergy2D(bendingEnergyDivisor);  
    }
  else if (NDimensions == 3)
    {
      bendingEnergy =  this->GetBendingEnergy3DMarc(bendingEnergyDivisor);  
    } 
  else 
    {
      itkExceptionMacro(<<"Wrong number of dimensions, this class only supports 2D or 3D transforms");
      return 0;
    } 
    
  niftkitkDebugMacro(<< "GetBendingEnergy():Finished, returning:" << bendingEnergy);

  m_BendingEnergyHasBeenUpdatedFlag = true;
  return bendingEnergy;
    
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
typename BSplineTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>::MeasureType
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::GetBendingEnergy2D(TScalarType divisor) const
{
  niftkitkDebugMacro(<< "GetBendingEnergy2D():Started");

  TScalarType tmp;
  TScalarType bendingEnergy = 0;
  TScalarType totalBendingEnergy = 0;
  
  GridIndexType             gridIndex;             // We iterate over every grid point.
  GridIndexType             gridMovingIndex;       // This is the moving index, as we iterate over neighbourhood.  
  GridPixelType             controlPointDisplacementVector; 
  GridSizeType              gridSize = m_Grid->GetLargestPossibleRegion().GetSize();
  
  // We iterate over the whole grid.      
  GridConstIteratorType gridIterator(this->m_Grid, this->m_Grid->GetLargestPossibleRegion());
  BendingEnergyIteratorType bendingEnergyIterator(this->m_BendingEnergyGrid, this->m_BendingEnergyGrid->GetLargestPossibleRegion());
  
  // These are indexes, to loop over control points.
  unsigned int l = 0;  
  unsigned int m = 0;   

  // These are zero, first and second derivatives of BSpline function w.r.t l, m
  TScalarType B_l, B_m, B_l_first, B_m_first, B_l_second, B_m_second;
  
  // These are for storing x and y components of derivative.
  TScalarType ll_x, ll_y, mm_x, mm_y;
  TScalarType lm_x, lm_y;
  
  gridIterator.GoToBegin();
  bendingEnergyIterator.GoToBegin();
  while (!gridIterator.IsAtEnd() && !bendingEnergyIterator.IsAtEnd())
    {
      gridIndex = gridIterator.GetIndex();
      
      B_l = B_m = B_l_first = B_m_first = B_l_second = B_m_second = 0.0;
      ll_x = ll_y = mm_x = mm_y = 0.0;
      lm_x = lm_y = 0.0;
      
      for (l = 0; l < 4; l++)
        {
          gridMovingIndex[0] = gridIndex[0] + l - 1;
          
          if (gridMovingIndex[0] >= (int) 0 && gridMovingIndex[0]  < (int) gridSize[0])
            {
            
              B_l        =  this->m_Lookup[0][l];
              B_l_first  =  this->m_Lookup1stDerivative[0][l];
              B_l_second =  this->m_Lookup2ndDerivative[0][l];
              
              for (m = 0; m < 4; m++)
                {
                  gridMovingIndex[1] = gridIndex[1] + m - 1;
                  
                  if (gridMovingIndex[1] >= (int) 0 && gridMovingIndex[1]  < (int) gridSize[1])
                    {
                    
                      B_m        =  this->m_Lookup[0][m];
                      B_m_first  =  this->m_Lookup1stDerivative[0][m];
                      B_m_second =  this->m_Lookup2ndDerivative[0][m];
                      
                      controlPointDisplacementVector = m_Grid->GetPixel(gridMovingIndex);
                              
                      tmp = B_l_second * B_m_first;
                      ll_x += controlPointDisplacementVector[0] * tmp;
                      ll_y += controlPointDisplacementVector[1] * tmp;
                              
                      tmp = B_m_second * B_l_first;
                      mm_x += controlPointDisplacementVector[0] * tmp;
                      mm_y += controlPointDisplacementVector[1] * tmp;
                              
                      tmp = B_l_first * B_m_first;
                      lm_x += controlPointDisplacementVector[0] * tmp;
                      lm_y += controlPointDisplacementVector[1] * tmp;

/*
                      niftkitkDebugMacro(<< "GetBendingEnergy2D():ll_x:" << ll_x \
                        << ", ll_y:" << ll_y \
                        << ", mm_x:" << mm_x \
                        << ", mm_y:" << mm_y \
                        << ", lm_x:" << lm_x \
                        << ", lm_y:" << lm_y \
                        << ", B_l:" << B_l \
                        << ", B_l_1:" << B_l_first \
                        << ", B_l_2:" << B_l_second \
                        << ", B_m:" << B_m \
                        << ", B_m_1:" << B_m_first \
                        << ", B_m_2:" << B_m_second \
                        << ", controlPointDisplacementVector:" << controlPointDisplacementVector);
*/
                    } // end if 
                } // end for m
            } // end if 
        } // end for l
      
      bendingEnergy =           (ll_x * ll_x) + (ll_y * ll_y)
                       +         (mm_x * mm_x) + (mm_y * mm_y)
                       + 2.0 * (
                                  (lm_x * lm_x) + (lm_y * lm_y)
                               );
      bendingEnergy /= divisor;
      totalBendingEnergy += bendingEnergy;
      bendingEnergyIterator.Set(bendingEnergy);
      
      // Dont forget these!
      ++gridIterator;
      ++bendingEnergyIterator;
      
    } // end for

  niftkitkDebugMacro(<< "GetBendingEnergy2D():Finished, returning:" << totalBendingEnergy);
  return totalBendingEnergy;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
typename BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>::MeasureType
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::GetBendingEnergy3DDaniel(TScalarType divisor) const
{
  niftkitkDebugMacro(<< "GetBendingEnergy3DDaniel():Started");
  
  TScalarType tmp;
  TScalarType bendingEnergy = 0;
  TScalarType totalBendingEnergy = 0;
  
  GridIndexType             gridIndex;             // We iterate over every grid point.
  GridIndexType             gridMovingIndex;       // This is the moving index, as we iterate over neighbourhood.
  GridPixelType             controlPointDisplacementVector; 
  GridSizeType              gridSize = m_Grid->GetLargestPossibleRegion().GetSize();
  
  // We iterate over the whole grid.      
  GridConstIteratorType gridIterator(this->m_Grid, this->m_Grid->GetLargestPossibleRegion());
  BendingEnergyIteratorType bendingEnergyIterator(this->m_BendingEnergyGrid, this->m_BendingEnergyGrid->GetLargestPossibleRegion());
  
  // These are indexes, to loop over control points.
  unsigned int l = 0;  
  unsigned int m = 0;  
  unsigned int n = 0;  

  // These are zero, first and second derivatives of BSpline function w.r.t l, m, n
  TScalarType B_l, B_m, B_n, B_l_first, B_m_first, B_n_first, B_l_second, B_m_second, B_n_second;
  
  // These are for storing x, y and z components of derivative.
  TScalarType ll_x, ll_y, ll_z, mm_x, mm_y, mm_z, nn_x, nn_y, nn_z;
  TScalarType lm_x, lm_y, lm_z, ln_x, ln_y, ln_z, mn_x, mn_y, mn_z;
  
  gridIterator.GoToBegin();
  bendingEnergyIterator.GoToBegin();
  
  while(!gridIterator.IsAtEnd() && !bendingEnergyIterator.IsAtEnd())
    {
      gridIndex = gridIterator.GetIndex();
      
      B_l = B_m = B_n = B_l_first = B_m_first = B_n_first = B_l_second = B_m_second = B_n_second = 0.0;
      ll_x = ll_y = ll_z = mm_x = mm_y = mm_z = nn_x = nn_y = nn_z = 0.0;
      lm_x = lm_y = lm_z = ln_x = ln_y = ln_z = mn_x = mn_y = mn_z = 0.0;
      
      for (l = 0; l < 4; l++)
        {
          gridMovingIndex[0] = gridIndex[0] + l - 1;
          
          if (gridMovingIndex[0] >= (int) 0 && gridMovingIndex[0]  < (int) gridSize[0])
            {
            
              B_l        =  this->m_Lookup[0][l];
              B_l_first  =  this->m_Lookup1stDerivative[0][l];
              B_l_second =  this->m_Lookup2ndDerivative[0][l];
              
              for (m = 0; m < 4; m++)
                {
                  gridMovingIndex[1] = gridIndex[1] + m - 1;
                  
                  if (gridMovingIndex[1] >= (int) 0 && gridMovingIndex[1]  < (int) gridSize[1])
                    {
                    
                      B_m        =  this->m_Lookup[0][m];
                      B_m_first  =  this->m_Lookup1stDerivative[0][m];
                      B_m_second =  this->m_Lookup2ndDerivative[0][m];
                      
                      for (n = 0; n < 4; n++)
                        {
                          gridMovingIndex[2] = gridIndex[2] + n - 1;
                          
                          if (gridMovingIndex[2] >= (int) 0 && gridMovingIndex[2]  < (int) gridSize[2])
                            {
                            
                              B_n        =  this->m_Lookup[0][n];
                              B_n_first  =  this->m_Lookup1stDerivative[0][n];
                              B_n_second =  this->m_Lookup2ndDerivative[0][n];
                            
                              controlPointDisplacementVector = m_Grid->GetPixel(gridMovingIndex);
                              
                              tmp = B_l_second * B_m_first * B_n_first;
                              ll_x += controlPointDisplacementVector[0] * tmp;
                              ll_y += controlPointDisplacementVector[1] * tmp;
                              ll_z += controlPointDisplacementVector[2] * tmp;
                              
                              tmp = B_m_second * B_l_first * B_n_first;
                              mm_x += controlPointDisplacementVector[0] * tmp;
                              mm_y += controlPointDisplacementVector[1] * tmp;
                              mm_z += controlPointDisplacementVector[2] * tmp;
                              
                              tmp = B_n_second * B_m_first * B_l_first;
                              nn_x += controlPointDisplacementVector[0] * tmp;
                              nn_y += controlPointDisplacementVector[1] * tmp;
                              nn_z += controlPointDisplacementVector[2] * tmp;
                              
                              tmp = B_l_first * B_m_first * B_n;
                              lm_x += controlPointDisplacementVector[0] * tmp;
                              lm_y += controlPointDisplacementVector[1] * tmp;
                              lm_z += controlPointDisplacementVector[2] * tmp;
                              
                              tmp = B_l_first * B_n_first * B_m;
                              ln_x += controlPointDisplacementVector[0] * tmp;
                              ln_y += controlPointDisplacementVector[1] * tmp;
                              ln_z += controlPointDisplacementVector[2] * tmp;

                              tmp = B_m_first * B_n_first * B_l;
                              mn_x += controlPointDisplacementVector[0] * tmp;
                              mn_y += controlPointDisplacementVector[1] * tmp;
                              mn_z += controlPointDisplacementVector[2] * tmp;
                              
                            } // end if 
                        } // end for n
                    } // end if
                } // end for m
            } // end if 
        } // end for l
      
      bendingEnergy  =           (ll_x * ll_x) + (ll_y * ll_y) + (ll_z * ll_z)
                       +         (mm_x * mm_x) + (mm_y * mm_y) + (mm_z * mm_z)
                       +         (nn_x * nn_x) + (nn_y * nn_y) + (nn_z * nn_z)
                       + 2.0 * (
                                  (lm_x * lm_x) + (lm_y * lm_y) + (lm_z * lm_z)
                                + (ln_x * ln_x) + (ln_y * ln_y) + (ln_z * ln_z)
                                + (mn_x * mn_x) + (mn_y * mn_y) + (mn_z * mn_z)
                               );
      bendingEnergy /= divisor;
      totalBendingEnergy += bendingEnergy;
      bendingEnergyIterator.Set(bendingEnergy);
      
      // Don't forget these!
      ++gridIterator; 
      ++bendingEnergyIterator;
      
    } // end for

  niftkitkDebugMacro(<< "GetBendingEnergy3DDaniel():Finished, returning:" << totalBendingEnergy);
  return totalBendingEnergy;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
typename BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>::MeasureType
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::GetBendingEnergy3DMarc(TScalarType divisor) const
{
  niftkitkDebugMacro(<< "GetBendingEnergy3DMarc():Started");
  
  TScalarType     totalBendingEnergy = 0; 
  GridSizeType    gSize              = m_Grid->GetLargestPossibleRegion().GetSize();
  GridSpacingType gSpacing           = this->m_Grid->GetSpacing();
  GridPixelType   controlPointDisplacementVector;
  
#if _USE_SSE
  union u{
  __m128 m;
  float f[4];
  } val;
#endif
  
  struct float3{
    float x,y,z;
  };  
  
  int nodeIndex = 0;
  
  // Copy grid into float3
  float3 *grid = new float3[gSize[0]*gSize[1]*gSize[2]];
  GridConstIteratorType gridIterator(this->m_Grid, this->m_Grid->GetLargestPossibleRegion());
  gridIterator.GoToBegin();
  while(!gridIterator.IsAtEnd())
    {
      controlPointDisplacementVector = gridIterator.Get();
      grid[nodeIndex].x = controlPointDisplacementVector[0];
      grid[nodeIndex].y = controlPointDisplacementVector[1];
      grid[nodeIndex].z = controlPointDisplacementVector[2];
      ++nodeIndex;
      ++gridIterator;
    }

  float3 gridRealSpacing;
  gridRealSpacing.x = gSpacing[0];
  gridRealSpacing.y = gSpacing[1];
  gridRealSpacing.z = gSpacing[2];
  
  float3 gridSize;
  gridSize.x = gSize[0];
  gridSize.y = gSize[1];
  gridSize.z = gSize[2];
  
  // As the contraint is only computed at the voxel position, the basis value of the spline are always the same 
  float normal[4];
  float first[4];
  float second[4];
  normal[0] = 1.0/6.0;
  normal[1] = 2.0/3.0;
  normal[2] = 1.0/6.0;
  normal[3] = 0.0;
  first[0] = -0.5;
  first[1] = 0.0;
  first[2] = 0.5;
  first[3] = 0.0;
  second[0] = 1.0;
  second[1] = -2.0;
  second[2] = 1.0;
  second[3] = 0.0;
  
  // There are six different values taken into account
  float tempXX[16];
  float tempYY[16];
  float tempZZ[16];
  float tempXY[16];
  float tempYZ[16];
  float tempXZ[16];
  
  int coord=0;
  for(int c=0; c<4; c++){
    for(int b=0; b<4; b++){
      tempXX[coord]=normal[c]*normal[b];  // z * y
      tempYY[coord]=normal[c]*second[b];  // z * y"
      tempZZ[coord]=second[c]*normal[b];  // z" * y
      tempXY[coord]=normal[c]*first[b]; // z * y'
      tempYZ[coord]=first[c]*first[b];  // z' * y'
      tempXZ[coord]=first[c]*normal[b]; // z' * y
      coord++;
    }
  }

  float basisXX[64];
  float basisYY[64];
  float basisZZ[64];
  float basisXY[64];
  float basisYZ[64];
  float basisXZ[64];
  
  coord=0;
  for(int bc=0; bc<16; bc++){
    for(int a=0; a<4; a++){
      basisXX[coord]=tempXX[bc]*second[a];  // z * y * x"
      basisYY[coord]=tempYY[bc]*normal[a];  // z * y" * x
      basisZZ[coord]=tempZZ[bc]*normal[a];  // z" * y * x
      basisXY[coord]=tempXY[bc]*first[a]; // z * y' * x'
      basisYZ[coord]=tempYZ[bc]*normal[a];  // z' * y' * x
      basisXZ[coord]=tempXZ[bc]*first[a]; // z' * y * x'
      coord++;
    }
  }
  
  float xControlPointCoordinates[64] ;
  float yControlPointCoordinates[64] ;
  float zControlPointCoordinates[64] ;
    
  float constraintValue=0.0;
  
  for(int z=0;z<gridSize.z;z++){
    for(int y=0;y<gridSize.y;y++){
      for(int x=0;x<gridSize.x;x++){
        
        coord=0;
        for(int Z=z-1; Z<z+3; Z++){
          for(int Y=y-1; Y<y+3; Y++){
            nodeIndex = (int)((Z*gridSize.y+Y)*gridSize.x+x-1);
            for(int X=x-1; X<x+3; X++){
              xControlPointCoordinates[coord] = (float)X * gridRealSpacing.x;
              yControlPointCoordinates[coord] = (float)Y * gridRealSpacing.y;
              zControlPointCoordinates[coord] = (float)Z * gridRealSpacing.z;
              if(-1<X && -1<Y && -1<Z && X<gridSize.x && Y<gridSize.y && Z<gridSize.z){
                xControlPointCoordinates[coord] -= grid[nodeIndex].x;
                yControlPointCoordinates[coord] -= grid[nodeIndex].y;
                zControlPointCoordinates[coord] -= grid[nodeIndex].z;
              }
              nodeIndex++;
              coord++;
            }
          }
        }
        
        float XX_x=0.0;
        float YY_x=0.0;
        float ZZ_x=0.0;
        float XY_x=0.0; 
        float YZ_x=0.0;
        float XZ_x=0.0;
        float XX_y=0.0;
        float YY_y=0.0;
        float ZZ_y=0.0;
        float XY_y=0.0;
        float YZ_y=0.0;
        float XZ_y=0.0;
        float XX_z=0.0;
        float YY_z=0.0;
        float ZZ_z=0.0;
        float XY_z=0.0;
        float YZ_z=0.0;
        float XZ_z=0.0;
#if _USE_SSE

        __m128 tempA =  _mm_set_ps1(0.0);
        __m128 tempB =  _mm_set_ps1(0.0);
        __m128 tempC =  _mm_set_ps1(0.0);
        __m128 tempD =  _mm_set_ps1(0.0);
        __m128 tempE =  _mm_set_ps1(0.0);
        __m128 tempF =  _mm_set_ps1(0.0);
        
        __m128 tempG =  _mm_set_ps1(0.0);
        __m128 tempH =  _mm_set_ps1(0.0);
        __m128 tempI =  _mm_set_ps1(0.0);
        __m128 tempJ =  _mm_set_ps1(0.0);
        __m128 tempK =  _mm_set_ps1(0.0);
        __m128 tempL =  _mm_set_ps1(0.0);
        
        __m128 tempM =  _mm_set_ps1(0.0);
        __m128 tempN =  _mm_set_ps1(0.0);
        __m128 tempO =  _mm_set_ps1(0.0);
        __m128 tempP =  _mm_set_ps1(0.0);
        __m128 tempQ =  _mm_set_ps1(0.0);
        __m128 tempR =  _mm_set_ps1(0.0);
        
        __m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
        __m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
        __m128 *ptrZ = (__m128 *) &zControlPointCoordinates[0];
        
        __m128 *ptrBasisXX   = (__m128 *) &basisXX[0];
        __m128 *ptrBasisYY   = (__m128 *) &basisYY[0];
        __m128 *ptrBasisZZ   = (__m128 *) &basisZZ[0];
        __m128 *ptrBasisXY   = (__m128 *) &basisXY[0];
        __m128 *ptrBasisYZ   = (__m128 *) &basisYZ[0];
        __m128 *ptrBasisXZ   = (__m128 *) &basisXZ[0];

        for(int a=0; a<16; ++a){
          tempA = _mm_add_ps(_mm_mul_ps(*ptrBasisXX, *ptrX), tempA );
          tempB = _mm_add_ps(_mm_mul_ps(*ptrBasisYY, *ptrX), tempB );
          tempC = _mm_add_ps(_mm_mul_ps(*ptrBasisZZ, *ptrX), tempC );
          tempD = _mm_add_ps(_mm_mul_ps(*ptrBasisXY, *ptrX), tempD );
          tempE = _mm_add_ps(_mm_mul_ps(*ptrBasisYZ, *ptrX), tempE );
          tempF = _mm_add_ps(_mm_mul_ps(*ptrBasisXZ, *ptrX), tempF );

          tempG = _mm_add_ps(_mm_mul_ps(*ptrBasisXX, *ptrY), tempG );
          tempH = _mm_add_ps(_mm_mul_ps(*ptrBasisYY, *ptrY), tempH );
          tempI = _mm_add_ps(_mm_mul_ps(*ptrBasisZZ, *ptrY), tempI );
          tempJ = _mm_add_ps(_mm_mul_ps(*ptrBasisXY, *ptrY), tempJ );
          tempK = _mm_add_ps(_mm_mul_ps(*ptrBasisYZ, *ptrY), tempK );
          tempL = _mm_add_ps(_mm_mul_ps(*ptrBasisXZ, *ptrY), tempL );

          tempM = _mm_add_ps(_mm_mul_ps(*ptrBasisXX, *ptrZ), tempM );
          tempN = _mm_add_ps(_mm_mul_ps(*ptrBasisYY, *ptrZ), tempN );
          tempO = _mm_add_ps(_mm_mul_ps(*ptrBasisZZ, *ptrZ), tempO );
          tempP = _mm_add_ps(_mm_mul_ps(*ptrBasisXY, *ptrZ), tempP );
          tempQ = _mm_add_ps(_mm_mul_ps(*ptrBasisYZ, *ptrZ), tempQ );
          tempR = _mm_add_ps(_mm_mul_ps(*ptrBasisXZ, *ptrZ), tempR );
          ++ptrBasisXX;
          ++ptrBasisYY;
          ++ptrBasisZZ;
          ++ptrBasisXY;
          ++ptrBasisYZ;
          ++ptrBasisXZ;
          ++ptrX;
          ++ptrY;
          ++ptrZ;
        }
        val.m=tempA;XX_x=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempB;YY_x=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempC;ZZ_x=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempD;XY_x=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempE;YZ_x=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempF;XZ_x=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        
        val.m=tempG;XX_y=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempH;YY_y=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempI;ZZ_y=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempJ;XY_y=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempK;YZ_y=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempL;XZ_y=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        
        val.m=tempM;XX_z=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempN;YY_z=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempO;ZZ_z=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempP;XY_z=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempQ;YZ_z=val.f[0]+val.f[1]+val.f[2]+val.f[3];
        val.m=tempR;XZ_z=val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
        for(int a=0; a<64; a++){
          XX_x += basisXX[a]*xControlPointCoordinates[a];
          YY_x += basisYY[a]*xControlPointCoordinates[a];
          ZZ_x += basisZZ[a]*xControlPointCoordinates[a];
          XY_x += basisXY[a]*xControlPointCoordinates[a];
          YZ_x += basisYZ[a]*xControlPointCoordinates[a];
          XZ_x += basisXZ[a]*xControlPointCoordinates[a];
                  
          XX_y += basisXX[a]*yControlPointCoordinates[a];
          YY_y += basisYY[a]*yControlPointCoordinates[a];
          ZZ_y += basisZZ[a]*yControlPointCoordinates[a];
          XY_y += basisXY[a]*yControlPointCoordinates[a];
          YZ_y += basisYZ[a]*yControlPointCoordinates[a];
          XZ_y += basisXZ[a]*yControlPointCoordinates[a];
                  
          XX_z += basisXX[a]*zControlPointCoordinates[a];
          YY_z += basisYY[a]*zControlPointCoordinates[a];
          ZZ_z += basisZZ[a]*zControlPointCoordinates[a];
          XY_z += basisXY[a]*zControlPointCoordinates[a];
          YZ_z += basisYZ[a]*zControlPointCoordinates[a];
          XZ_z += basisXZ[a]*zControlPointCoordinates[a];
        }
#endif
        
        constraintValue += XX_x*XX_x + YY_x*YY_x + ZZ_x*ZZ_x + 2.0*(XY_x*XY_x + YZ_x*YZ_x + XZ_x*XZ_x);
        constraintValue += XX_y*XX_y + YY_y*YY_y + ZZ_y*ZZ_y + 2.0*(XY_y*XY_y + YZ_y*YZ_y + XZ_y*XZ_y);
        constraintValue += XX_z*XX_z + YY_z*YY_z + ZZ_z*ZZ_z + 2.0*(XY_z*XY_z + YZ_z*YZ_z + XZ_z*XZ_z);
      }
    }
  }

  totalBendingEnergy = constraintValue/(3.0*gridSize.x*gridSize.y*gridSize.z);
  
  niftkitkDebugMacro(<< "GetBendingEnergy3DMarc():Finished, returning:" << totalBendingEnergy);
  delete[] grid;
  
  return totalBendingEnergy;
}

  
template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
TScalarType 
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::GetNewOrigin(TScalarType oldSize, TScalarType oldSpacing, TScalarType oldOrigin, TScalarType newSize, TScalarType newSpacing)
{
  TScalarType result = 0;//((oldSize-1)*oldSpacing/2.0 + oldOrigin) - ((newSize-1)*newSpacing/2.0);
  return result;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::InterpolateNextGrid(FixedImagePointer image)
{
  niftkitkDebugMacro(<< "InterpolateNextGrid():Starting");

      niftkitkDebugMacro(<< "InterpolateNextGrid():BEFORE min def=" << niftk::ConvertToString(this->ComputeMinDeformation()) \
          << ", max def=" << niftk::ConvertToString(this->ComputeMaxDeformation()) \
          << ", min jac=" << niftk::ConvertToString(this->ComputeMinJacobian()) \
          << ", max jac=" << niftk::ConvertToString(this->ComputeMaxJacobian()));

  GridRegionType oldGridRegion = m_Grid->GetLargestPossibleRegion();
  GridSizeType oldGridSize = oldGridRegion.GetSize();
  GridIndexType oldGridIndex = oldGridRegion.GetIndex();
  GridSpacingType oldGridSpacing = m_Grid->GetSpacing();
  GridDirectionType oldGridDirection = m_Grid->GetDirection();
  GridOriginType oldGridOrigin = m_Grid->GetOrigin();

  // Copy current grid to old, and blank newGrid.  
  this->m_OldGrid->SetRegions(oldGridRegion);
  this->m_OldGrid->SetSpacing(oldGridSpacing);
  this->m_OldGrid->SetDirection(oldGridDirection);
  this->m_OldGrid->SetOrigin(oldGridOrigin);
  this->m_OldGrid->Allocate();
  
  typedef itk::ImageRegionIterator<GridImageType> OldIteratorType;
  OldIteratorType oldIterator(this->m_OldGrid, this->m_OldGrid->GetLargestPossibleRegion());
  
  typedef itk::ImageRegionIterator<GridImageType> CurrentIteratorType;
  CurrentIteratorType currentIterator(this->m_Grid, this->m_Grid->GetLargestPossibleRegion());
  
  GridPixelType zero;
  zero.Fill(0);
  
  oldIterator.GoToBegin();
  currentIterator.GoToBegin();
  while(!currentIterator.IsAtEnd() && !oldIterator.IsAtEnd())
    {
      oldIterator.Set(currentIterator.Get());  
      currentIterator.Set(zero);
      
      ++oldIterator;
      ++currentIterator;
    }
  
  GridSizeType newGridSize;
  GridIndexType newGridIndex;
  GridSpacingType newGridSpacing;
  GridOriginType newGridOrigin;

  for (unsigned int i = 0; i < NDimensions; i++)
    {
      newGridIndex[i]   = 0;
      newGridSize[i]    = 2*oldGridSize[i] - 1;
      newGridSpacing[i] = oldGridSpacing[i]/2.0;
      newGridOrigin[i]  = GetNewOrigin(oldGridSize[i], oldGridSpacing[i], oldGridOrigin[i], newGridSize[i], newGridSpacing[i]);
    }
  
  GridDirectionType newGridDirection = oldGridDirection;  
  GridRegionType newGridRegion;
  newGridRegion.SetSize(newGridSize);
  newGridRegion.SetIndex(newGridIndex);
  
  // Resize new image, and bending energy image, which must be the same size.
  this->m_Grid->SetRegions(newGridRegion);
  this->m_Grid->SetSpacing(newGridSpacing);
  this->m_Grid->SetDirection(newGridDirection);
  this->m_Grid->SetOrigin(newGridOrigin);
  this->m_Grid->Allocate();
  this->m_Grid->Update();
  this->m_Grid->FillBuffer(zero);
  
  this->m_BendingEnergyGrid->SetRegions(newGridRegion);
  this->m_BendingEnergyGrid->SetSpacing(newGridSpacing);
  this->m_BendingEnergyGrid->SetDirection(newGridDirection);
  this->m_BendingEnergyGrid->SetOrigin(newGridOrigin);
  this->m_BendingEnergyGrid->Allocate();
  this->m_BendingEnergyGrid->Update();
  this->m_BendingEnergyGrid->FillBuffer(0);
  
  niftkitkDebugMacro(<< "InterpolateNextGrid():Changed from size:" << this->m_OldGrid->GetLargestPossibleRegion().GetSize() \
    << ", spacing:" << this->m_OldGrid->GetSpacing() \
    << ", origin:" << this->m_OldGrid->GetOrigin());

  niftkitkDebugMacro(<< "InterpolateNextGrid():        to size:" << this->m_Grid->GetLargestPossibleRegion().GetSize() \
    << ", spacing:" << this->m_Grid->GetSpacing() \
    << ", origin:" << this->m_Grid->GetOrigin());

  // now fill it up.
    
  if (NDimensions == 2)
    {
      InterpolateNextGrid2D(this->m_OldGrid, this->m_Grid);
    }
  else if (NDimensions == 3)
    {
      InterpolateNextGrid3D(this->m_OldGrid, this->m_Grid);
    }
  else
    {
      itkExceptionMacro(<<"Wrong number of dimensions, this class only supports 2D or 3D transforms");
      return;    
    }

  // And set parameters from grid, this forces a resize of the base class parameters array.
  this->SetParametersFromField(this->m_Grid, true);

  // This forces a resize of the base class deformation field, which also sets it to zero deformation.
  Superclass::Initialize(image);

  // So, we need to take the existing parameters, and update the deformation field.
  if (NDimensions == 2)
    {
      this->InterpolateDeformationField2D();  
    }
  else if (NDimensions == 3)
    {
      this->InterpolateDeformationField3DMarc();  
    } 
  else 
    {
      itkExceptionMacro(<<"Wrong number of dimensions, this class only supports 2D or 3D transforms");
    } 

      niftkitkDebugMacro(<< "InterpolateNextGrid():AFTER min def=" << niftk::ConvertToString(this->ComputeMinDeformation()) \
          << ", max def=" << niftk::ConvertToString(this->ComputeMaxDeformation()) \
          << ", min jac=" << niftk::ConvertToString(this->ComputeMinJacobian()) \
          << ", max jac=" << niftk::ConvertToString(this->ComputeMaxJacobian()));
  
  /*
  if (s_Logger->isDebugEnabled())
    {
      this->WriteControlPointImage("tmp.bspline.after.vtk");
    }
  */
  
  niftkitkDebugMacro(<< "InterpolateNextGrid():Finished");
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::InterpolateNextGrid2D(GridImagePointer& oldGrid, GridImagePointer &newGrid)
{
  unsigned int x, y, i1, j1, i2, j2;
  
  niftkitkDebugMacro(<< "InterpolateNextGrid2D():Started");
  
  TScalarType w[2][3];
  w[1][0] = 0;
  w[1][1] = 1.0/2.0;
  w[1][2] = 1.0/2.0;
  w[0][0] = 1.0/8.0;
  w[0][1] = 6.0/8.0;
  w[0][2] = 1.0/8.0;

  GridSizeType oldSize = oldGrid->GetLargestPossibleRegion().GetSize();
  GridSizeType newSize = newGrid->GetLargestPossibleRegion().GetSize();
  GridIndexType oldIndex;
  GridIndexType newIndex;
  GridPixelType oldPixel;
  GridPixelType newPixel;
  
  for (x = 0; x < oldSize[0]; x++)
    {
      for (y = 0; y < oldSize[1]; y++)
        {
          for (i1 = 0; i1 < 2; i1++)
            {
              for (j1 = 0; j1 < 2; j1++)
                {    
                  newIndex[0] = 2*x + i1;
                  newIndex[1] = 2*y + j1;

                  if (newIndex[0] >= 0 && (unsigned int)newIndex[0] < newSize[0]
                   && newIndex[1] >= 0 && (unsigned int)newIndex[1] < newSize[1])
                    {
                      newPixel[0] = 0;
                      newPixel[1] = 0;
                        
                      for (i2 = 0; i2 < 3; i2++)
                        {
                          for (j2 = 0; j2 < 3; j2++)
                            {
                              oldIndex[0] = x + i2 -1;
                              oldIndex[1] = y + j2 -1;

                              if (oldIndex[0] >= 0 && (unsigned int)oldIndex[0] < oldSize[0]
                               && oldIndex[1] >= 0 && (unsigned int)oldIndex[1] < oldSize[1])
                                {
                                  oldPixel = oldGrid->GetPixel(oldIndex);
                                  
                                  newPixel[0] += w[i1][i2] * w[j1][j2] * oldPixel[0];
                                  newPixel[1] += w[i1][i2] * w[j1][j2] * oldPixel[1];                                  
                                }
                            }
                        }                                
                      newGrid->SetPixel(newIndex, newPixel);                      
                    }
                }   
            }
        }
    }

  niftkitkDebugMacro(<< "InterpolateNextGrid2D():Finished");
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
BSplineTransform<TFixedImage, TScalarType,NDimensions,TDeformationScalar>
::InterpolateNextGrid3D(GridImagePointer& oldGrid, GridImagePointer &newGrid)
{
  unsigned int x, y, z, i1, j1, k1, i2, j2, k2;
  
  TScalarType w[2][3];  
  w[1][0] = 0;
  w[1][1] = 1.0/2.0;
  w[1][2] = 1.0/2.0;
  w[0][0] = 1.0/8.0;
  w[0][1] = 6.0/8.0;
  w[0][2] = 1.0/8.0;

  GridSizeType oldSize = oldGrid->GetLargestPossibleRegion().GetSize();
  GridSizeType newSize = newGrid->GetLargestPossibleRegion().GetSize();
  GridIndexType oldIndex;
  GridIndexType newIndex;
  GridPixelType oldPixel;
  GridPixelType newPixel;
  
  for (x = 0; x < oldSize[0]; x++)
    {
      for (y = 0; y < oldSize[1]; y++)
        {
          for (z = 0; z < oldSize[2]; z++)
            {
            
              for (i1 = 0; i1 < 2; i1++)
                {
                  for (j1 = 0; j1 < 2; j1++)
                    {
                      for (k1 = 0; k1 < 2; k1++)
                        {
        
                          newIndex[0] = 2*x + i1;
                          newIndex[1] = 2*y + j1;
                          newIndex[2] = 2*z + k1;

                          if (newIndex[0] >= 0 && (unsigned int)newIndex[0] < newSize[0]
                           && newIndex[1] >= 0 && (unsigned int)newIndex[1] < newSize[1]
                           && newIndex[2] >= 0 && (unsigned int)newIndex[2] < newSize[2])
                            {
                              
                              newPixel[0] = 0;
                              newPixel[1] = 0;
                              newPixel[2] = 0;
                              
                              for (i2 = 0; i2 < 3; i2++)
                                {
                                  for (j2 = 0; j2 < 3; j2++)
                                    {
                                      for (k2 = 0; k2 < 3; k2++)
                                        {
                                        
                                          oldIndex[0] = x + i2 -1;
                                          oldIndex[1] = y + j2 -1;
                                          oldIndex[2] = z + k2 -1;
                                          
                                          if (oldIndex[0] >= 0 && (unsigned int)oldIndex[0] < oldSize[0]
                                           && oldIndex[1] >= 0 && (unsigned int)oldIndex[1] < oldSize[1]
                                           && oldIndex[2] >= 0 && (unsigned int)oldIndex[2] < oldSize[2])
                                            {
                                              oldPixel = oldGrid->GetPixel(oldIndex);
                                 
                                              newPixel[0] += w[i1][i2] * w[j1][j2] * w[k1][k2] * oldPixel[0];
                                              newPixel[1] += w[i1][i2] * w[j1][j2] * w[k1][k2] * oldPixel[1];
                                              newPixel[2] += w[i1][i2] * w[j1][j2] * w[k1][k2] * oldPixel[2];
                                              
                                            }
                                        }
                                    }
                                }
                        
                              newGrid->SetPixel(newIndex, newPixel);
                              
                            }
                        }
                    }
                }
            }
        }
    }
}
template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::WriteControlPointImage(std::string filename)
{
  niftkitkDebugMacro(<< "WriteControlPointImage():filename:" << filename);
  
  typedef float OutputVectorDataType;
  typedef Vector<OutputVectorDataType, NDimensions> OutputVectorPixelType;
  typedef Image<OutputVectorPixelType, NDimensions> OutputVectorImageType;
  typedef CastImageFilter<GridImageType, OutputVectorImageType> CastFilterType;
  typedef ImageFileWriter<OutputVectorImageType> WriterType;
  
  typename CastFilterType::Pointer caster = CastFilterType::New();
  typename WriterType::Pointer writer = WriterType::New();

  caster->SetInput(this->m_Grid);
  writer->SetFileName(filename);
  writer->SetInput(caster->GetOutput());
  writer->Update();
  
  niftkitkDebugMacro(<< "WriteControlPointImage():Done:" << filename);
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::GetBendingEnergyDerivativeDaniel(DerivativeType & derivative)
{
  niftkitkDebugMacro(<< "GetBendingEnergyDerivativeDaniel():Started");
  
  this->m_BendingEnergyDerivativeFilter->SetInput(const_cast<BendingEnergyImageConstPointer>(m_BendingEnergyGrid.GetPointer()));
  this->m_BendingEnergyDerivativeFilter->SetNormalize(false);
  this->m_BendingEnergyDerivativeFilter->Modified();
  this->m_BendingEnergyDerivativeFilter->UpdateLargestPossibleRegion();

  unsigned long int parameterIndex = 0;
  BendingEnergyDerivativePixelType value;
  
  BendingEnergyDerivativeIteratorType iterator(m_BendingEnergyDerivativeFilter->GetOutput(), m_BendingEnergyDerivativeFilter->GetOutput()->GetLargestPossibleRegion());
  
  iterator.GoToBegin();
  while(!iterator.IsAtEnd())
    {
      value = iterator.Get();
      for (unsigned int dimension = 0; dimension < NDimensions; dimension++)
        {
          derivative.SetElement(parameterIndex, value[dimension]);
          parameterIndex++;
        }
      ++iterator;
    }

  niftkitkDebugMacro(<< "GetBendingEnergyDerivativeDaniel():Finished");
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::GetBendingEnergyDerivativeMarc(DerivativeType & derivative)
{
  niftkitkDebugMacro(<< "GetBendingEnergyDerivativeMarc():Started");

  GridSizeType    gSize = m_Grid->GetLargestPossibleRegion().GetSize();
  GridSpacingType gSpacing = this->m_Grid->GetSpacing();
  GridPixelType   controlPointDisplacementVector; 
  unsigned long int parameterIndex = 0;
  
  // The constraint gradient is only computed at the node position (approximation) in order to save memory space and time.
  // Now we compute the constraint gradient for every control point and add the value to the NMI gradient
  struct float3{
    float x,y,z;
  };  

  float normal[4];
  float first[4];
  float second[4];
  normal[0] = 1.0/6.0;
  normal[1] = 2.0/3.0;
  normal[2] = 1.0/6.0;
  normal[3] = 0.0;
  first[0] = -0.5f;
  first[1] = 0.0;
  first[2] = 0.5f;
  first[3] = 0.0;
  second[0] = 1.0;
  second[1] = -2.0;
  second[2] = 1.0;
  second[3] = 0.0;

  // There are six different values taken into account
  float tempXX[16];
  float tempYY[16];
  float tempZZ[16];
  float tempXY[16];
  float tempYZ[16];
  float tempXZ[16];

  int coord=0;
  for(int c=0; c<4; c++){
    for(int b=0; b<4; b++){
      tempXX[coord]=normal[c]*normal[b];  // z  * y
      tempYY[coord]=normal[c]*second[b];  // z  * y"
      tempZZ[coord]=second[c]*normal[b];  // z" * y
      tempXY[coord]=normal[c]*first[b]; // z  * y'
      tempYZ[coord]=first[c]*first[b];  // z' * y'
      tempXZ[coord]=first[c]*normal[b]; // z' * y
      coord++;
    }
  }

  float basisXX[64] ;
  float basisYY[64] ;
  float basisZZ[64] ;
  float basisXY[64] ;
  float basisYZ[64] ;
  float basisXZ[64] ;

  coord=0;
  for(int bc=0; bc<16; bc++){
    for(int a=0; a<4; a++){
      basisXX[coord]=tempXX[bc]*second[a];  // z  * y  * x"
      basisYY[coord]=tempYY[bc]*normal[a];  // z  * y" * x
      basisZZ[coord]=tempZZ[bc]*normal[a];  // z" * y  * x
      basisXY[coord]=tempXY[bc]*first[a]; // z  * y' * x'
      basisYZ[coord]=tempYZ[bc]*normal[a];  // z' * y' * x
      basisXZ[coord]=tempXZ[bc]*first[a]; // z' * y  * x'
      coord++;
    }
  }

  int nodeIndex = 0;
  
  float xControlPointCoordinates[64] ;
  float yControlPointCoordinates[64] ;
  float zControlPointCoordinates[64] ;

  float3 gridSize;
  gridSize.x = gSize[0];
  gridSize.y = gSize[1];
  gridSize.z = gSize[2];

  float3 gridRealSpacing;
  gridRealSpacing.x = gSpacing[0];
  gridRealSpacing.y = gSpacing[1];
  gridRealSpacing.z = gSpacing[2];

  int nodeNumber = (int)(gridSize.x*gridSize.y*gridSize.z);
  float3 *derivativeValues = new float3[6*nodeNumber];
  float3 *grid = new float3[nodeNumber];
  
  GridConstIteratorType gridIterator(this->m_Grid, this->m_Grid->GetLargestPossibleRegion());
  gridIterator.GoToBegin();
  while(!gridIterator.IsAtEnd())
    {
      controlPointDisplacementVector = gridIterator.Get();
      grid[nodeIndex].x = controlPointDisplacementVector[0];
      grid[nodeIndex].y = controlPointDisplacementVector[1];
      grid[nodeIndex].z = controlPointDisplacementVector[2];
      ++nodeIndex;
      ++gridIterator;
    }
  
  nodeIndex = 0;
  
  for(int z=0;z<gridSize.z;z++){
    for(int y=0;y<gridSize.y;y++){
      for(int x=0;x<gridSize.x;x++){

        coord=0;
        for(int Z=z-1; Z<z+3; Z++){
          for(int Y=y-1; Y<y+3; Y++){
            int gridIndex = (int)((Z*gridSize.y+Y)*gridSize.x+x-1);
            for(int X=x-1; X<x+3; X++){
              xControlPointCoordinates[coord] = (float)X * gridRealSpacing.x;
              yControlPointCoordinates[coord] = (float)Y * gridRealSpacing.y;
              zControlPointCoordinates[coord] = (float)Z * gridRealSpacing.z;
              if(-1<X && -1<Y && -1<Z && X<gridSize.x && Y<gridSize.y && Z<gridSize.z){
                xControlPointCoordinates[coord] -= grid[gridIndex].x;
                yControlPointCoordinates[coord] -= grid[gridIndex].y;
                zControlPointCoordinates[coord] -= grid[gridIndex].z;
              }
              gridIndex++;
              coord++;
            }
          }
        }

        float3 XX;
        float3 YY;
        float3 ZZ;
        float3 XY;
        float3 YZ;
        float3 XZ;
        XX.x=XX.y=XX.z=0.0;
        YY.x=YY.y=YY.z=0.0;
        ZZ.x=ZZ.y=ZZ.z=0.0;
        XY.x=XY.y=XY.z=0.0;
        YZ.x=YZ.y=YZ.z=0.0;
        XZ.x=XZ.y=XZ.z=0.0;

        for(int a=0; a<64; a++){
          XX.x += basisXX[a]*xControlPointCoordinates[a];
          YY.x += basisYY[a]*xControlPointCoordinates[a];
          ZZ.x += basisZZ[a]*xControlPointCoordinates[a];

          XX.y += basisXX[a]*yControlPointCoordinates[a];
          YY.y += basisYY[a]*yControlPointCoordinates[a];
          ZZ.y += basisZZ[a]*yControlPointCoordinates[a];

          XX.z += basisXX[a]*zControlPointCoordinates[a];
          YY.z += basisYY[a]*zControlPointCoordinates[a];
          ZZ.z += basisZZ[a]*zControlPointCoordinates[a];
          
          XY.x += basisXY[a]*xControlPointCoordinates[a];
          YZ.x += basisYZ[a]*xControlPointCoordinates[a];
          XZ.x += basisXZ[a]*xControlPointCoordinates[a];

          XY.y += basisXY[a]*yControlPointCoordinates[a];
          YZ.y += basisYZ[a]*yControlPointCoordinates[a];
          XZ.y += basisXZ[a]*yControlPointCoordinates[a];

          XY.z += basisXY[a]*zControlPointCoordinates[a];
          YZ.z += basisYZ[a]*zControlPointCoordinates[a];
          XZ.z += basisXZ[a]*zControlPointCoordinates[a];
        }

        XX.x*=2.0;XX.y*=2.0;XX.z*=2.0;
        YY.x*=2.0;YY.y*=2.0;YY.z*=2.0;
        ZZ.x*=2.0;ZZ.y*=2.0;ZZ.z*=2.0;
        XY.x*=4.0;XY.y*=4.0;XY.z*=4.0;
        YZ.x*=4.0;YZ.y*=4.0;YZ.z*=4.0;
        XZ.x*=4.0;XZ.y*=4.0;XZ.z*=4.0;
        derivativeValues[nodeIndex]=XX;
        derivativeValues[nodeIndex+nodeNumber]=YY;
        derivativeValues[nodeIndex+nodeNumber*2]=ZZ;
        derivativeValues[nodeIndex+nodeNumber*3]=XY;
        derivativeValues[nodeIndex+nodeNumber*4]=YZ;
        derivativeValues[nodeIndex+nodeNumber*5]=XZ;

        nodeIndex++;
      }
    }
  }

  nodeIndex = 0;
  for(int z=0;z<gridSize.z;z++){
    for(int y=0;y<gridSize.y;y++){
      for(int x=0;x<gridSize.x;x++){

        float3 gradientValue;
        gradientValue.x=gradientValue.y=gradientValue.z=0.0;
  
        coord=0;
        for(int Z=z-1; Z<z+3; Z++){
          for(int Y=y-1; Y<y+3; Y++){
            int indexXX = (int)((Z*gridSize.y+Y)*gridSize.x+x-1);
            int indexYY = indexXX+nodeNumber;
            int indexZZ = indexYY+nodeNumber;
            int indexXY = indexZZ+nodeNumber;
            int indexYZ = indexXY+nodeNumber;
            int indexXZ = indexYZ+nodeNumber;
            for(int X=x-1; X<x+3; X++){
              if(-1<X && -1<Y && -1<Z && X<gridSize.x && Y<gridSize.y && Z<gridSize.z){
                gradientValue.x += derivativeValues[indexXX].x*basisXX[coord];
                gradientValue.y += derivativeValues[indexXX].y*basisXX[coord];
                gradientValue.z += derivativeValues[indexXX].z*basisXX[coord];

                gradientValue.x += derivativeValues[indexYY].x*basisYY[coord];
                gradientValue.y += derivativeValues[indexYY].y*basisYY[coord];
                gradientValue.z += derivativeValues[indexYY].z*basisYY[coord];

                gradientValue.x += derivativeValues[indexZZ].x*basisZZ[coord];
                gradientValue.y += derivativeValues[indexZZ].y*basisZZ[coord];
                gradientValue.z += derivativeValues[indexZZ].z*basisZZ[coord];

                gradientValue.x += derivativeValues[indexXY].x*basisXY[coord];
                gradientValue.y += derivativeValues[indexXY].y*basisXY[coord];
                gradientValue.z += derivativeValues[indexXY].z*basisXY[coord];

                gradientValue.x += derivativeValues[indexYZ].x*basisYZ[coord];
                gradientValue.y += derivativeValues[indexYZ].y*basisYZ[coord];
                gradientValue.z += derivativeValues[indexYZ].z*basisYZ[coord];

                gradientValue.x += derivativeValues[indexXZ].x*basisXZ[coord];
                gradientValue.y += derivativeValues[indexXZ].y*basisXZ[coord];
                gradientValue.z += derivativeValues[indexXZ].z*basisXZ[coord];
              }
              indexXX++;
              indexYY++;
              indexZZ++;
              indexXY++;
              indexYZ++;
              indexXZ++;
              coord++;
            }
          }
        }

        derivative.SetElement(parameterIndex++, gradientValue.x/(float)nodeNumber);
        derivative.SetElement(parameterIndex++, gradientValue.y/(float)nodeNumber);
        derivative.SetElement(parameterIndex++, gradientValue.z/(float)nodeNumber);
        
        //printf("Matt\tconstGradient %d = %f, %f, %f\n", nodeIndex, gradientValue.x, gradientValue.y, gradientValue.z);
        nodeIndex++;
      }
    }
  }

  delete[] derivativeValues;
  delete[] grid;

  niftkitkDebugMacro(<< "GetBendingEnergyDerivativeMarc():Finished");
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
BSplineTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::GetBendingEnergyDerivative(DerivativeType & derivative)
{
  niftkitkDebugMacro(<< "GetBendingEnergyDerivative():Started with:" << derivative.GetSize() << " parameters");
  
  if (derivative.GetSize() != this->m_Parameters.GetSize())
    {
      itkExceptionMacro(<<"The parameters array has:" << this->m_Parameters.GetSize() << " elements, but the derivative array has:" << derivative.GetSize() << ", so the derivative array is the wrong size");           
    }

  if (NDimensions == 2)
    {
      this->GetBendingEnergyDerivativeDaniel(derivative);  
    }
  else if (NDimensions == 3)
    {
      this->GetBendingEnergyDerivativeMarc(derivative);  
    } 

  niftkitkDebugMacro(<< "GetBendingEnergyDerivative():Finished with:" << derivative.GetSize() << " parameters");
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
const typename BSplineTransform<TFixedImage, TScalarType, NDimensions,TDeformationScalar>::ParametersType& 
BSplineTransform<TFixedImage, TScalarType,NDimensions,TDeformationScalar>
::GetFixedParameters(void) const 
{
  // Get a copy of Fixed parameters from base class.
  ParametersType baseClassFixedParams = Superclass::GetFixedParameters();
  int baseClassSize = baseClassFixedParams.GetSize();
  
  // Resize the Fixed parameters and copy in the base class ones
  // as these pertain to the deformation field.
  int index = 0;
  this->m_FixedParameters.SetSize(2*(baseClassSize));
  this->m_FixedParameters.Fill(0);
  for (index = 0; index < baseClassSize; index++)
    {
      this->m_FixedParameters[index] = baseClassFixedParams[index];
    }
   
  // Now we output similar information for the internal BSpline grid.
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    this->m_FixedParameters.SetElement(index, this->m_Grid->GetSpacing()[i]);
    index++; 
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    for (unsigned int j = 0; j < NDimensions; j++)
    {
      this->m_FixedParameters.SetElement(index, this->m_Grid->GetDirection()[i][j]);
      index++; 
    }
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    this->m_FixedParameters.SetElement(index, this->m_Grid->GetOrigin()[i]);
    index++; 
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    this->m_FixedParameters.SetElement(index, this->m_Grid->GetLargestPossibleRegion().GetSize()[i]);
    index++; 
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    this->m_FixedParameters.SetElement(index, this->m_Grid->GetLargestPossibleRegion().GetIndex()[i]);
    index++; 
  }
  return this->m_FixedParameters; 
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void 
BSplineTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::SetFixedParameters(const ParametersType& parameters)
{
  // First we collect the parameters for the deformation field.
  typename FixedImageType::Pointer tempFixedImage = FixedImageType::New(); 
  DeformationFieldSpacingType spacing;
  DeformationFieldDirectionType direction;
  DeformationFieldOriginType origin;
  DeformationFieldSizeType regionSize;
  DeformationFieldIndexType regionIndex;
  DeformationFieldRegionType region;
  int index = 0; 
  
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    spacing[i] = parameters.GetElement(index);
    index++; 
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    for (unsigned int j = 0; j < NDimensions; j++)
    {
      direction[i][j] = parameters.GetElement(index);
      index++; 
    }
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    origin[i] = parameters.GetElement(index);
    index++; 
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    regionSize[i] = (unsigned long int)parameters.GetElement(index);
    index++; 
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    regionIndex[i] = (long int)parameters.GetElement(index);
    index++; 
  }
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);
  
  tempFixedImage->SetSpacing(spacing);
  tempFixedImage->SetDirection(direction);
  tempFixedImage->SetOrigin(origin);
  tempFixedImage->SetRegions(region); 

  // Then we collect the parameters for the BSpline grid
  GridRegionType    gridRegion;
  GridIndexType     gridIndex;
  GridSizeType      gridSize;
  GridSpacingType   gridSpacing;
  GridDirectionType gridDirection;
  GridOriginType    gridOrigin;

  for (unsigned int i = 0; i < NDimensions; i++)
  {
    gridSpacing[i] = parameters.GetElement(index);
    index++; 
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    for (unsigned int j = 0; j < NDimensions; j++)
    {
      gridDirection[i][j] = parameters.GetElement(index);
      index++; 
    }
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    gridOrigin[i] = parameters.GetElement(index);
    index++; 
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    gridSize[i] = (unsigned long int)parameters.GetElement(index);
    index++; 
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    gridIndex[i] = (long int)parameters.GetElement(index);
    index++; 
  }
  gridRegion.SetSize(gridSize);
  gridRegion.SetIndex(gridIndex);

  this->InitializeGrid(tempFixedImage.GetPointer(), gridRegion, gridSpacing, gridDirection, gridOrigin); 
}


} // namespace

#endif // __itkBSplineTransform_txx

