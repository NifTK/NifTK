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
#ifndef __itkJorgesInitializationRelaxStreamlinesFilter_h
#define __itkJorgesInitializationRelaxStreamlinesFilter_h

#include "itkLagrangianInitializedRelaxStreamlinesFilter.h"

namespace itk {
/** 
 * \class JorgesInitializationRelaxStreamlinesFilter
 * \brief Implementation of Jorges method to initialize GM boundaries prior to cortical thickness calculations. 
 *  
 * \sa RelaxStreamlinesFilter
 */
template < class TImageType, typename TScalarType, unsigned int NDimensions> 
class ITK_EXPORT JorgesInitializationRelaxStreamlinesFilter :
  public RelaxStreamlinesFilter< TImageType, TScalarType, NDimensions>
{
public:

  /** Standard "Self" typedef. */
  typedef JorgesInitializationRelaxStreamlinesFilter                    Self;
  typedef RelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>  Superclass;
  typedef SmartPointer<Self>                                            Pointer;
  typedef SmartPointer<const Self>                                      ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(JorgesInitializationRelaxStreamlinesFilter, RelaxStreamlinesFilter);

  /** Standard typedefs. */
  typedef typename Superclass::InputVectorImagePixelType      InputVectorImagePixelType;
  typedef typename Superclass::InputVectorImageType           InputVectorImageType;
  typedef typename Superclass::InputVectorImagePointer        InputVectorImagePointer;
  typedef typename Superclass::InputVectorImageConstPointer   InputVectorImageConstPointer;
  typedef typename Superclass::InputVectorImageIndexType      InputVectorImageIndexType; 
  typedef typename Superclass::InputScalarImagePixelType      InputScalarImagePixelType;
  typedef typename Superclass::InputScalarImageType           InputScalarImageType;
  typedef typename Superclass::InputScalarImagePointType      InputScalarImagePointType;
  typedef typename Superclass::InputScalarImagePointer        InputScalarImagePointer;
  typedef typename Superclass::InputScalarImageIndexType      InputScalarImageIndexType;
  typedef typename Superclass::InputScalarImageConstPointer   InputScalarImageConstPointer;
  typedef typename Superclass::InputScalarImageRegionType     InputScalarImageRegionType;
  typedef typename Superclass::InputScalarImageSpacingType    InputScalarImageSpacingType;
  typedef typename InputScalarImageType::SizeType             InputScalarImageSizeType;
  typedef typename InputScalarImageType::PointType            InputScalarImageOriginType;
  typedef typename Superclass::OutputImageType                OutputImageType;
  typedef typename Superclass::OutputImagePixelType           OutputImagePixelType;
  typedef typename Superclass::OutputImagePointer             OutputImagePointer;
  typedef typename Superclass::OutputImageConstPointer        OutputImageConstPointer;
  typedef typename Superclass::OutputImageIndexType           OutputImageIndexType;
  typedef typename Superclass::OutputImageSpacingType         OutputImageSpacingType;
  typedef typename OutputImageType::RegionType                OutputImageRegionType;
  typedef typename OutputImageType::SizeType                  OutputImageSizeType;
  typedef typename OutputImageType::DirectionType             OutputImageDirectionType;
  typedef typename OutputImageType::PointType                 OutputImageOriginType;  
  typedef typename Superclass::VectorInterpolatorType         VectorInterpolatorType;
  typedef typename Superclass::VectorInterpolatorPointer      VectorInterpolatorPointer;
  typedef typename Superclass::VectorInterpolatorPointType    VectorInterpolatorPointType;
  typedef typename Superclass::ScalarInterpolatorType         ScalarInterpolatorType;
  typedef typename Superclass::ScalarInterpolatorPointer      ScalarInterpolatorPointer;
  typedef typename Superclass::ScalarInterpolatorPointType    ScalarInterpolatorPointType;
  typedef ContinuousIndex<TScalarType, TImageType::ImageDimension> ContinuousIndexType;
  typedef Point<TScalarType, TImageType::ImageDimension> PointType;

  /** Sets the segmented/label image, at input 2. */
  void SetSegmentedImage(const InputScalarImageType *image) {this->SetNthInput(2, const_cast<InputScalarImageType *>(image)); }

  /** Sets the pv map, at input 3. */
  void SetGMPVMap(const InputScalarImageType *image) {this->SetNthInput(3, const_cast<InputScalarImageType *>(image)); }

protected:
  JorgesInitializationRelaxStreamlinesFilter();
  ~JorgesInitializationRelaxStreamlinesFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;
  
private:
  
  /**
   * Prohibited copy and assignment. 
   */
  JorgesInitializationRelaxStreamlinesFilter(const Self&); 
  void operator=(const Self&); 
  
  OutputImagePixelType GetInitilizedValue(
      InputScalarImagePixelType& segmentedValue,
      InputScalarImageType* segmentedImage, 
      InputScalarImageType* gmPVImage, 
      InputScalarImageIndexType& index, 
      bool& initializeLOImage, 
      bool& initializeL1Image, 
      bool& addToGreyList);
  
  void InitializeBoundaries(
      std::vector<InputScalarImageIndexType>& completeListOfGreyMatterPixels,
      InputScalarImageType* scalarImage,
      InputVectorImageType* vectorImage,
      OutputImageType* L0Image,
      OutputImageType* L1Image,
      std::vector<InputScalarImageIndexType>& L0greyList,
      std::vector<InputScalarImageIndexType>& L1greyList
      );
  
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkJorgesInitializationRelaxStreamlinesFilter.txx"
#endif

#endif
