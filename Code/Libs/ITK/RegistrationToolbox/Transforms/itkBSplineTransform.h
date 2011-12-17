/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkBSplineTransform_h
#define __itkBSplineTransform_h

#include <iostream>
#include "itkDeformableTransform.h"
#include "itkImage.h"
#include "itkImageRegion.h"
#include "itkVector.h"
#include "itkArray2D.h"

#include "itkContinuousIndex.h"
#include "itkImageRegionIterator.h"
#include "itkSingleValuedCostFunction.h"
#include "itkScalarImageToNormalizedGradientVectorImageFilter.h"

namespace itk
{

/** 
 * \class BSplineTransform
 * \brief Deformable transform using a BSpline representation.
 * 
 * SetParameters and GetParameters are used by the standard ITK optimizers.
 * You should call Initialize() with your fixed image, which will set up the 
 * size of the deformation field, according to the grid spacing.  
 * 
 * Alternatively, you can force the grid size/space/origin using SetGridSize,
 * SetGridSpacing, SetGridOrigin. When you call these methods, the parameters 
 * array is resized, and reset.
 *
 * Internally, Set/GetGridSize, Set/GetGridSpacing, Set/GetGridOrigin operate
 * directly onto the control point grid.
 * 
 * \ingroup Transforms
 */
template <
    class TFixedImage,                   // Templated over the image type.
    class TScalarType,                   // Data type for scalars
    unsigned int NDimensions,            // Number of Dimensions i.e. 2D or 3D
    class TDeformationScalar             // Data type in the deformation field. 
    >            
class ITK_EXPORT BSplineTransform : 
          public DeformableTransform< TFixedImage, TScalarType, NDimensions, TDeformationScalar >
{
public:
  
  /** Standard class typedefs. */
  typedef BSplineTransform                                            Self;
  typedef DeformableTransform< TFixedImage, TScalarType, NDimensions, TDeformationScalar> Superclass;
  typedef SmartPointer<Self>                                          Pointer;
  typedef SmartPointer<const Self>                                    ConstPointer;
      
  /** New macro for creation of through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( BSplineTransform, DeformableTransform );

  /** Get the number of dimensions. */
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);

  /** Standard scalar type for this class. */
  typedef typename Superclass::ScalarType                         ScalarType;

  /** Standard parameters container. */
  typedef typename Superclass::ParametersType                     ParametersType;
  typedef typename Superclass::DerivativeType                     DerivativeType;
  typedef typename Superclass::JacobianType                       JacobianType;

  /** Standard coordinate point type for this class. */
  typedef typename Superclass::InputPointType                     InputPointType;
  typedef typename Superclass::OutputPointType                    OutputPointType;

  /** Typedefs for the deformation field. */
  typedef typename Superclass::DeformationFieldPixelType          DeformationFieldPixelType;
  typedef typename Superclass::DeformationFieldType               DeformationFieldType;
  typedef typename Superclass::DeformationFieldPointer            DeformationFieldPointer;
  typedef typename Superclass::DeformationFieldRegionType         DeformationFieldRegionType;
  typedef typename Superclass::DeformationFieldIndexType          DeformationFieldIndexType;
  typedef typename Superclass::DeformationFieldSizeType           DeformationFieldSizeType;
  typedef typename Superclass::DeformationFieldSpacingType        DeformationFieldSpacingType;
  typedef typename Superclass::DeformationFieldDirectionType      DeformationFieldDirectionType;
  typedef typename Superclass::DeformationFieldOriginType         DeformationFieldOriginType;
  
  /** Typedefs for the grid of control points. */
  typedef Vector< TDeformationScalar, NDimensions >               GridPixelType;
  typedef Image<GridPixelType, NDimensions>                       GridImageType;
  typedef typename GridImageType::Pointer                         GridImagePointer;
  typedef ImageRegion<NDimensions>                                GridRegionType;
  typedef typename GridRegionType::IndexType                      GridIndexType;
  typedef typename GridRegionType::SizeType                       GridSizeType;
  typedef typename GridImageType::SpacingType                     GridSpacingType;
  typedef typename GridImageType::DirectionType                   GridDirectionType;
  typedef typename GridImageType::PointType                       GridOriginType;
  
  /** typedefs for image of bending energy. */      
  typedef TDeformationScalar                                      BendingEnergyPixelType;
  typedef Image<TDeformationScalar, NDimensions>                  BendingEnergyImageType;
  typedef ImageRegion<NDimensions>                                BendingEnergyImageRegionType;
  typedef typename BendingEnergyImageRegionType::SizeType         BendingEnergyImageSizeType;
  typedef typename BendingEnergyImageType::Pointer                BendingEnergyImagePointer;
  typedef const BendingEnergyImageType*                           BendingEnergyImageConstPointer;  
  typedef ImageRegionIterator<BendingEnergyImageType>             BendingEnergyIteratorType;
  typedef ScalarImageToNormalizedGradientVectorImageFilter<BendingEnergyImageType, 
                                                     TDeformationScalar> BendingEnergyDerivativeFilterType;
  typedef typename BendingEnergyDerivativeFilterType::Pointer     BendingEnergyDerivativeFilterPointer;
  typedef typename BendingEnergyDerivativeFilterType::OutputPixelType BendingEnergyDerivativePixelType;
  typedef typename BendingEnergyDerivativeFilterType::OutputImageType BendingEnergyDerivativeImageType;
  typedef ImageRegionConstIterator<BendingEnergyDerivativeImageType>  BendingEnergyDerivativeIteratorType;
  
  /** The deformation field is defined over the fixed image. */
  typedef TFixedImage                                             FixedImageType;
  typedef typename FixedImageType::ConstPointer                   FixedImagePointer;
  
  /** Typedefs for internal stuff. */
  typedef itk::ContinuousIndex<TDeformationScalar, NDimensions>   GridVoxelCoordinateType;
  typedef itk::ImageRegionIterator<DeformationFieldType>          DeformationFieldIteratorType;    
  typedef itk::ImageRegionConstIteratorWithIndex<GridImageType>   GridConstIteratorType; 
  typedef itk::ImageRegionIterator<GridImageType>                 GridIteratorType;
  
  /** For returning the bending energy. */
  typedef SingleValuedCostFunction::MeasureType                   MeasureType;

  /** 
   * Set the deformation field to Identity, (no deformation).
   * Doesn't affect the Global transform.
   * Doesn't resize anything.
   */
  virtual void SetIdentity();

  /** 
   * Convenience method to set up internal images.
   * Sets the internal deformation field to the same size as fixed image.
   * Sets the grid size, and parameters array to the right size grid, and calls SetIdentity().
   */
  void Initialize(FixedImagePointer fixedImage, GridSpacingType finalGridSpacingInMillimetres, int numberOfLevels);
  
  /** Increases the resolution of the grid, for when you are doing multi-resolution. */
  void InterpolateNextGrid(FixedImagePointer fixedImage);
  
  /** 
   * This method sets the parameters of the transform. For a BSpline 
   * deformation transform, the parameters are the control point displacements.
   * 
   * Imagine an N-D grid, with the same configuration as an N-D image
   * (ie. same, x,y,z order.. with x increasing most quickly and z
   * increasing least quickly.). At each point, you have 3 cooefficients
   * which represent a vector at that point. So, the order of the
   * parameters are the 3 cooefficients (x,y,z) for each point in turn.
   *
   */
  void SetParameters(const ParametersType & parameters);

  /** Get the fixed parameters for saving. */
  virtual const ParametersType& GetFixedParameters(void) const;
  
  /** Set the fixed paramters for loading */
  virtual void SetFixedParameters(const ParametersType& parameters);

  /** Returns the bending energy of the transformation. */
  MeasureType GetBendingEnergy();

  /** This defaults to false, and will go to true, if bending energy is ever evaluated. */
  itkGetMacro(BendingEnergyHasBeenUpdatedFlag, bool);
  
  /** The writes the derivatives into the supplied array. */
  void GetBendingEnergyDerivative(DerivativeType & derivative);
  
  /** Get bending energy derivative in a method like Daniel Rueckerts. */
  void GetBendingEnergyDerivativeDaniel(DerivativeType & derivative);
  
  /** Get bending energy derivative like Marc Modat. */
  void GetBendingEnergyDerivativeMarc(DerivativeType & derivative);
  
  /** Access to the Grid of control points. */
  itkGetObjectMacro(Grid, GridImageType);

  /** Write parameters, so subclass can override if necessary. So when this is called on 
   * base class, we write control points instead of the deformation field. */
  virtual void WriteParameters(std::string filename) { this->WriteControlPointImage(filename); }

  /** Write out the current transformation as an image of vectors. */
  void WriteControlPointImage(std::string filename);

  /** Declared virtual in base class, transform points*/
  virtual OutputPointType  TransformPoint(const InputPointType  &point ) const;

protected:

  BSplineTransform();
  virtual ~BSplineTransform();

  /** Print contents of an BSplineDeformableTransform. */
  void PrintSelf(std::ostream &os, Indent indent) const;

private:
  BSplineTransform(const Self&); // purposely not implemented
  void operator=(const Self&);   // purposely not implemented

  /** Size of lookup table for BSpline weights. */
  const static unsigned int s_LookupTableRows = 1000;
  const static unsigned int s_LookupTableSize = s_LookupTableRows - 1;
  const static unsigned int s_LookupTableCols = 4;
  
  /** An grid of control point offsets.*/
  GridImagePointer          m_Grid;
  GridImagePointer          m_OldGrid;
  
  /** For bending energy related stuff. */
  bool                                 m_BendingEnergyHasBeenUpdatedFlag;
  BendingEnergyImagePointer            m_BendingEnergyGrid;
  BendingEnergyDerivativeFilterPointer m_BendingEnergyDerivativeFilter;
  
  /** Fills in the deformation field for 2D. */
  void InterpolateDeformationField2D();
  
  /** Fills in the deformation field for 3D. */
  void InterpolateDeformationField3DDaniel();

  /** Fills in the deformation field for 3D. */
  void InterpolateDeformationField3DMarc();

  /** Calculates bending energy for 2D. */
  MeasureType GetBendingEnergy2D(TScalarType divisor) const;
  
  /** Calculates bending energy for 3D. */
  MeasureType GetBendingEnergy3DDaniel(TScalarType divisor) const;

  /** Calculates bending energy for 3D. */
  MeasureType GetBendingEnergy3DMarc(TScalarType divisor) const;

  /** Interpolate next grid 2D. */
  void InterpolateNextGrid2D(GridImagePointer& oldGrid, GridImagePointer &newGrid);
  
  /** Interpolate next grid 3D. */
  void InterpolateNextGrid3D(GridImagePointer& oldGrid, GridImagePointer &newGrid);
  
  /** Calculates where the origin of the grid should be. */
  TScalarType GetNewOrigin(TScalarType oldSize, TScalarType oldSpacing, TScalarType oldOrigin, TScalarType newSize, TScalarType newSpacing);
  
  /** Internal method that actually sets stuff up. */
  void InitializeGrid(FixedImagePointer fixedImage, 
      GridRegionType gridRegion,
      GridSpacingType gridSpacing,
      GridDirectionType gridDirection,
      GridOriginType gridOrigin);
  
  /** Lookup tables. */
  Array2D<TScalarType>    m_Lookup;
  Array2D<TScalarType>    m_Lookup1stDerivative;
  Array2D<TScalarType>    m_Lookup2ndDerivative;

  TScalarType B0(TScalarType u)  { return ((1-u)*(1-u)*(1-u))/6.0; } 
  TScalarType B1(TScalarType u)  { return (3*u*u*u - 6*u*u + 4)/6.0; }
  TScalarType B2(TScalarType u)  { return (-3*u*u*u + 3*u*u + 3*u + 1)/6.0; }
  TScalarType B3(TScalarType u)  { return (u*u*u)/6.0; }
  TScalarType B01(TScalarType u) { return -(1-u)*(1-u)/2.0; }
  TScalarType B11(TScalarType u) { return (9*u*u - 12*u)/6.0; }
  TScalarType B21(TScalarType u) { return (-9*u*u + 6*u + 3)/6.0; }
  TScalarType B31(TScalarType u) { return (u*u)/2.0; }
  TScalarType B02(TScalarType u) { return 1-u; }
  TScalarType B12(TScalarType u) { return 3*u - 2; }
  TScalarType B22(TScalarType u) { return -3*u + 1; }
  TScalarType B32(TScalarType u) { return u; }

  TScalarType B(int i, TScalarType u)
    {
      switch (i) {
        case 0: return B0(u);
        case 1: return B1(u);
        case 2: return B2(u);
        case 3: return B3(u);
      }
      return 0;
    }

  TScalarType B1(int i, TScalarType u)
    {
      switch (i) {
        case 0: return B01(u);
        case 1: return B11(u);
        case 2: return B21(u);
        case 3: return B31(u);
      }
      return 0;
    }

  TScalarType B2(int i, TScalarType u)
    {
      switch (i) {
        case 0: return B02(u);
        case 1: return B12(u);
        case 2: return B22(u);
        case 3: return B32(u);
      }
      return 0;
    }

  TScalarType B(int i, int derivative, TScalarType u)
    {
      switch(derivative) {
        case 0: return B(i, u);
        case 1: return B1(i, u);
        case 2: return B2(i, u);
      }
      return 0;
    }
  
}; // class BSplineTransform


}  // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBSplineTransform.txx"
#endif

#endif /* __itkBSplineTransform_h */
