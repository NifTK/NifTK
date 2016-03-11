/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramMLOorCCViewCalculator_h
#define __itkMammogramMLOorCCViewCalculator_h

#include <itkImage.h>
#include <itkObject.h>
#include <itkMacro.h>
#include <itkImageLinearConstIteratorWithIndex.h>
#include <itkImageRegionIterator.h>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>

namespace itk
{

/** \class MammogramMLOorCCViewCalculator
 *  \brief Computes whether a mammogram corresponds to a CC or an MLO view
 */

template < class TInputImage >
class ITK_EXPORT MammogramMLOorCCViewCalculator 
  : public Object
{

public:

  /** Standard class typedefs. */
  typedef MammogramMLOorCCViewCalculator Self;
  typedef Object                        Superclass;
  typedef SmartPointer< Self >          Pointer;
  typedef SmartPointer< const Self >    ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MammogramMLOorCCViewCalculator, Object);

  /** Image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension);

  /** Type definition for the input image. */
  typedef TInputImage InputImageType;

  typedef typename InputImageType::Pointer      ImagePointer;
  typedef typename InputImageType::ConstPointer ImageConstPointer;
  typedef typename InputImageType::PixelType    InputImagePixelType;
  typedef typename InputImageType::IndexType    InputImageIndexType;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef typename InputImageType::SizeType     InputImageSizeType;
  typedef typename InputImageType::PointType    InputImagePointType;
  typedef typename InputImageType::SpacingType  InputImageSpacingType;

  typedef typename itk::ImageLinearConstIteratorWithIndex< InputImageType > LineIteratorType;
  typedef typename itk::ImageRegionConstIterator< InputImageType > IteratorType;

  typedef itk::MetaDataDictionary DictionaryType;
  typedef itk::MetaDataObject< std::string > MetaDataStringType;

  /// Breast view
  typedef enum {
    UNKNOWN_MAMMO_VIEW,
    CC_MAMMO_VIEW,
    MLO_MAMMO_VIEW
  } MammogramViewType;

  /** Connect the input image. */
  void SetImage( const InputImageType *imInput );

  /** If a dictionary is supplied the this will be scanned for the strings 'MLO' or 'CC'. */
  void SetDictionary( DictionaryType &dictionary ) { m_Dictionary = dictionary; }

  /** If the image file name is supplied the this will be scanned for the strings 'MLO' or 'CC'. */
  void SetImageFileName( std::string fileName ) { m_ImageFileName = fileName; }

  /** Compute which breast was imaged in the mammogram. */
  void Compute(void) throw (ExceptionObject);

  /** Return the threshold intensity value. */
  itkGetConstMacro(MammogramView, MammogramViewType);

  /** Return the normalised cross-correlation value. */
  double GetViewScore( void ) { return m_Score; }

  void SetVerbose( bool flag )  { m_FlgVerbose = flag; }
  void SetVerboseOn( void )  { m_FlgVerbose = true;  }
  void SetVerboseOff( void ) { m_FlgVerbose = false; }

protected:

  MammogramMLOorCCViewCalculator();
  virtual ~MammogramMLOorCCViewCalculator() { }

  bool m_FlgVerbose;

  InputImageRegionType  m_ImRegion;
  InputImageSpacingType m_ImSpacing;
  InputImagePointType   m_ImOrigin;
  InputImageSizeType    m_ImSize;
  InputImageIndexType   m_ImStart;
  InputImagePointType   m_ImSizeInMM;

  ImageConstPointer m_Image;

  std::string m_ImageFileName;

  DictionaryType m_Dictionary;

  MammogramViewType m_MammogramView;
  double m_Score;

  void PrintSelf(std::ostream & os, Indent indent) const;

private:

  MammogramMLOorCCViewCalculator(const Self &); //purposely not implemented
  void operator=(const Self &);                //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMammogramMLOorCCViewCalculator.txx"
#endif

#endif /* __itkMammogramMLOorCCViewCalculator_h */
