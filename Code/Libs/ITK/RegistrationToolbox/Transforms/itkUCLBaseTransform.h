/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkUCLBaseTransform_h
#define itkUCLBaseTransform_h

#include <itkTransform.h>

namespace itk
{
  
/** 
 * \class UCLBaseTransform
 * \brief Our base transform class. Cant think of a better name.
 *
 * \ingroup Transforms
 *
 */
template <class TScalarType,
          unsigned int NInputDimensions=3, 
          unsigned int NOutputDimensions=3>
class ITK_EXPORT  UCLBaseTransform  : public Transform<TScalarType, NInputDimensions, NOutputDimensions>
{
public:
  /** Standard class typedefs. */
  typedef UCLBaseTransform                                            Self;
  typedef Transform<TScalarType, NInputDimensions, NOutputDimensions> Superclass;
  typedef SmartPointer< Self >                                        Pointer;
  typedef SmartPointer< const Self >                                  ConstPointer;
  typedef typename Superclass::InputPointType                         InputPointType;
  typedef typename Superclass::OutputPointType                        OutputPointType;
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( UCLBaseTransform, Transform );

  /** To transform a point, without creating an intermediate one. */
  virtual void TransformPoint(const InputPointType  &input, OutputPointType &output ) const = 0; 
  
  /** To get the inverse. Returns false, if transform is non-invertable. */
  virtual bool GetInv(UCLBaseTransform* inverse) const = 0; 

protected:
  UCLBaseTransform() {}; 
  UCLBaseTransform(unsigned int Dimension, unsigned int NumberOfParameters) : Transform<TScalarType, NInputDimensions, NOutputDimensions>(Dimension, NumberOfParameters) {};
  virtual ~UCLBaseTransform() {}

private:
  UCLBaseTransform(const Self&);  // purposefully not implemented
  void operator=(const Self&); // purposefully not implemented

};

} // end namespace itk

#endif
