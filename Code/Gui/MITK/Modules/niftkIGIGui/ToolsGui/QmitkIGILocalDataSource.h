/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKIGILOCALDATASOURCE_H
#define QMITKIGILOCALDATASOURCE_H

#include "niftkIGIGuiExports.h"
#include "QmitkIGIDataSource.h"
#include <mitkITKImageImport.txx>
#include <itkImportImageFilter.h>
#include <itkRGBPixel.h>
#include <cv.h>

class QmitkIGILocalDataSourceGrabbingThread;

/**
 * \class QmitkIGILocalDataSource
 * \brief Base class for IGI Data Sources that are not receiving networked input,
 * and hence are grabbing data from the local machine - eg. Video grabber.
 */
class NIFTKIGIGUI_EXPORT QmitkIGILocalDataSource : public QmitkIGIDataSource
{

  Q_OBJECT

public:

  friend class QmitkIGILocalDataSourceGrabbingThread;

  mitkClassMacro(QmitkIGILocalDataSource, QmitkIGIDataSource);

  typedef itk::RGBPixel< unsigned char >  UCRGBPixelType;
  typedef itk::Image< UCRGBPixelType, 2 >  RGBItkImage;
  typedef itk::ImportImageFilter< UCRGBPixelType, 2 >  RGBImportFilterType;

  typedef itk::RGBAPixel< unsigned char > UCRGBAPixelType;
  typedef itk::Image< UCRGBAPixelType, 2 > RGBAItkImage;
  typedef itk::ImportImageFilter< UCRGBAPixelType, 2 > RGBAImportFilterType;

protected:

  QmitkIGILocalDataSource(mitk::DataStorage* storage); // Purposefully hidden.
  virtual ~QmitkIGILocalDataSource(); // Purposefully hidden.

  QmitkIGILocalDataSource(const QmitkIGILocalDataSource&); // Purposefully not implemented.
  QmitkIGILocalDataSource& operator=(const QmitkIGILocalDataSource&); // Purposefully not implemented.


  mitk::Image::Pointer CreateMitkImage(const IplImage* image) const;

  /**
   * \brief Helper method for sub-classes, that will instantiate a
   * new MITK image from an RGB OpenCV IplImage.
   */
  mitk::Image::Pointer CreateRGBMitkImage(const IplImage* image) const;
  mitk::Image::Pointer CreateRGBAMitkImage(const IplImage* image) const;

  /**
   * \brief Derived classes call this when they are ready for the updates to start,
   * and this method instantiates the thread and laumches it.
   */
  void InitializeAndRunGrabbingThread(const int& intervalInMilliseconds);

  /**
   * \brief Called by QmitkIGILocalDataSourceGrabbingThread.
   */
  virtual void GrabData() = 0;

protected:
  QmitkIGILocalDataSourceGrabbingThread *m_GrabbingThread;

}; // end class

#endif

