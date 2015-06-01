/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkUltrasoundPinCalibrationWidget.h"
#include <stdexcept>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkFileIOUtils.h>
#include <niftkFileHelper.h>
#include <stdexcept>
#include <iostream>
#include <QApplication>
#include <QMessageBox>
#include <vtkPNGReader.h>
//#include <vtkNIFTIImageReader.h> // this header isn't in VTK yet

//-----------------------------------------------------------------------------
QmitkUltrasoundPinCalibrationWidget::QmitkUltrasoundPinCalibrationWidget(
  const QString& inputImageDirectory,
  const QString& outputPointDirectory,  
  QWidget *parent)
: QVTKWidget(parent)
, m_InputImageDirectory(inputImageDirectory)
, m_OutputPointDirectory(outputPointDirectory)
, m_PNG(true)
{
  m_ImageViewer = vtkImageViewer::New();
  this->SetRenderWindow(m_ImageViewer->GetRenderWindow());
  m_ImageViewer->SetupInteractor(this->GetRenderWindow()->GetInteractor());
  m_ImageViewer->SetColorLevel(127.5);
  m_ImageViewer->SetColorWindow(255);  
  
  if ( m_PNG )
  {
    m_ImageReader = dynamic_cast<vtkImageReader2*>(vtkPNGReader::New());
    // Load all data, and set up the PNG reader to the first image.
    m_ImageFiles = niftk::FindFilesWithGivenExtension(m_InputImageDirectory.toStdString(), ".png");
  }
  else
  {
    //m_ImageReader = dynamic_cast<vtkImageReader2*>(vtkNIFTIImageReader::New());
    // Load all data, and set up the NII reader to the first image.
    m_ImageFiles = niftk::FindFilesWithGivenExtension(m_InputImageDirectory.toStdString(), ".nii");
  }
  
  if (m_ImageFiles.size() == 0)
  {
    throw std::runtime_error("Did not find any .png files.");
  }
  std::sort(m_ImageFiles.begin(), m_ImageFiles.end());
  m_ImageFileCounter = 0;
  m_PointsOutputCounter = 0;

  m_ImageReader->SetFileName(m_ImageFiles[m_ImageFileCounter].c_str());
  m_ImageViewer->SetInputConnection(m_ImageReader->GetOutputPort());
  m_ImageReader->Update();

  int extent[6];
  m_ImageReader->GetDataExtent(extent);
  m_ImageWidth = extent[1] + 1;
  m_ImageHeight = extent[3] + 1;
}


//-----------------------------------------------------------------------------
QmitkUltrasoundPinCalibrationWidget::~QmitkUltrasoundPinCalibrationWidget()
{
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::enterEvent(QEvent* event)
{
  this->resize(m_ImageWidth, m_ImageHeight);
  QVTKWidget::enterEvent(event);
}

//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::SetPNG ( bool png ) 
{
  m_PNG = png;
}

//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::mousePressEvent(QMouseEvent* event)
{
  try 
  {
    this->StorePoint(event);
    event->accept();
  }
  catch (std::exception& e)
  {
    std::cerr << "Caught std::exception: " << e.what() << std::endl;
    event->ignore();
  }
  catch (...)
  {
    std::cerr << "Caught unknown exception: " << std::endl;
    event->ignore();
  }
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::keyPressEvent(QKeyEvent* event)
{
  if (event->key() == Qt::Key_N)
  {
    this->NextImage();
    event->accept();    
  }
  else if (event->key() == Qt::Key_P)
  {
    this->PreviousImage();
    event->accept();
  }
  else if (event->key() == Qt::Key_Q)
  {
    this->QuitApplication();
    event->accept();
  }
  else
  {
    event->ignore();
  }
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::QuitApplication()
{
  QApplication::exit(0);
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::ShowImage(const unsigned long int& imageNumber)
{
  m_ImageReader->SetFileName(m_ImageFiles[imageNumber].c_str());
  m_ImageViewer->Render();

  int offsetImageNumber = imageNumber + 1;
  std::cout << "Displaying image[" << offsetImageNumber << "/" << m_ImageFiles.size() << ", " << offsetImageNumber*100/m_ImageFiles.size() << "%]=" << m_ImageFiles[imageNumber] << ", stored " << m_PointsOutputCounter << " so far." << std::endl;
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::NextImage()
{
  if (m_ImageFileCounter < m_ImageFiles.size() - 1)
  {
    m_ImageFileCounter++;
    this->ShowImage(m_ImageFileCounter);
  }
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::PreviousImage()
{
  if (m_ImageFileCounter > 0)
  {
    m_ImageFileCounter--;
    this->ShowImage(m_ImageFileCounter);
  }
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::CreateDir(const std::string& dir)
{
  if (!niftk::DirectoryExists(dir))
  {
    if (!niftk::CreateDirAndParents(dir))
    {
      QMessageBox::critical(this, tr("niftkUltrasoundPinCalibrationSorter"),
                                  tr("Can't write to\n%1").arg(QString::fromStdString(dir)),
                                  QMessageBox::Ok);
      QApplication::exit(-1);
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkUltrasoundPinCalibrationWidget::StorePoint(QMouseEvent* event)
{
  if (event != NULL)
  {
    int xPixel = event->x();
    int yPixel = event->y();
    Qt::MouseButton button = event->button();
    double zCoordinate = 0;

    if (button == Qt::LeftButton)
    {
      this->CreateDir(m_OutputPointDirectory.toStdString());

      QString imageFileName = QString::fromStdString(m_ImageFiles[m_ImageFileCounter]);
      QRegExp rx("([0-9]{19})");

      int matchIndex = imageFileName.indexOf(rx);
      if (matchIndex != -1)
      {
        QString imageTimeStampString = imageFileName.mid(matchIndex,19);
        QString baseNameForPoint = imageTimeStampString + QString(".txt");
        std::string pointFileFullPath = niftk::ConvertToFullNativePath((m_OutputPointDirectory + QString("/") + baseNameForPoint).toStdString());

        ofstream myfile(pointFileFullPath.c_str(), std::ofstream::out | std::ofstream::trunc);
        if (myfile.is_open())
        {
          myfile << xPixel << " " << yPixel << " " << zCoordinate << std::endl;
          myfile.close();
          m_PointsOutputCounter++;
        }
        else
        {
          QMessageBox::warning(this, tr("niftkUltrasoundPinCalibrationSorter"),
                                      tr("Failed to write point to file\n%1").arg(QString::fromStdString(pointFileFullPath)),
                                      QMessageBox::Ok);
        }
      }
      else
      {
        QMessageBox::warning(this, tr("niftkUltrasoundPinCalibrationSorter"),
                                    tr("Invalid image file name\n%1").arg(imageFileName),
                                    QMessageBox::Ok);
      }
    }
  }
}

