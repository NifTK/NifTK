/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3326 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkUCLBaseTransform_h
#define __itkUCLBaseTransform_h

#include "itkTransform.h"

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
