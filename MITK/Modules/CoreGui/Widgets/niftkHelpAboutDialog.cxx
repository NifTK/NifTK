/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkHelpAboutDialog.h"

#include <QMessageBox>
#include <QString>
#include <iostream>
#include <NifTKConfigure.h>

namespace niftk
{

HelpAboutDialog::HelpAboutDialog(QWidget *parent, QString applicationName)
                : m_ApplicationName(applicationName)
{
  if(!m_ApplicationName.isEmpty())
  {
    this->setupUi(this);
    this->GenerateHelpAboutText(m_ApplicationName);
    m_HelpAboutLabel->setWordWrap(true);
    m_HelpAboutLabel->setOpenExternalLinks(true);
    m_HelpAboutLabel->setTextFormat(Qt::RichText);
  }
}

HelpAboutDialog::~HelpAboutDialog()
{

}

void HelpAboutDialog::GenerateHelpAboutText(QString applicationName)
{
  // This stuff gets generated during CMake into NifTKConfigure.h
  QString platformName(NIFTK_PLATFORM);
  QString versionNumber(NIFTK_VERSION_STRING);
  int versionNumberMajor(NIFTK_VERSION_MAJOR);
  int versionNumberMinor(NIFTK_VERSION_MINOR);
  int versionNumberPatch(NIFTK_VERSION_PATCH);
  QString copyrightText(NIFTK_COPYRIGHT);
  QString originURL(NIFTK_ORIGIN_URL);
  QString originShortText(NIFTK_ORIGIN_SHORT_TEXT);
  QString originLongText(NIFTK_ORIGIN_LONG_TEXT);
  QString documentationLocation(NIFTK_DOC_LOCATION
                                + QString("/v")
                                + QObject::tr("%1.").arg(versionNumberMajor)
                                + QObject::tr("%1.").arg(versionNumberMinor, 2, 10, QChar('0'))
                                + QObject::tr("%1").arg(versionNumberPatch)
                                );
  QString userContact(NIFTK_USER_CONTACT);
  QString qtVersion(NIFTK_QT_VERSION);
  QString boostVersion(NIFTK_BOOST_VERSION);
  QString gdcmVersion(NIFTK_GDCM_VERSION);
  QString dcmtkVersion(NIFTK_DCMTK_VERSION);
  QString itkVersion(NIFTK_ITK_VERSION);
  QString vtkVersion(NIFTK_VTK_VERSION);
  QString ctkVersion(NIFTK_CTK_VERSION);
  QString mitkVersion(NIFTK_MITK_VERSION);
  QString niftkVersion(NIFTK_VERSION);
  QString niftkDateTime(NIFTK_DATE_TIME);
  QString boostLocation(NIFTK_BOOST_LOCATION);
  QString gdcmLocation(NIFTK_GDCM_LOCATION);
  QString dcmtkLocation(NIFTK_DCMTK_LOCATION);
  QString itkLocation(NIFTK_ITK_LOCATION);
  QString vtkLocation(NIFTK_VTK_LOCATION);
  QString ctkLocation(NIFTK_CTK_LOCATION);
  QString mitkLocation(NIFTK_MITK_LOCATION);
  
  // Main titles with application name, release version and copyright statement.
  QString titles = QObject::tr(
      "<p>"
      "<h1>About %1 - %2</h1>"
      "(git hash %3, at %4, from <a href=\"https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK\">here</a>)"
      "</p>"
      "<p>%5 Please go to the installation folder or "
      "<a href=\"%6\">online documentation</a> for a full license description "
      "for this product and dependencies.</p>"
      )
      .arg(applicationName)
      .arg(versionNumber)
      .arg(niftkVersion)
      .arg(niftkDateTime)
      .arg(copyrightText)
      .arg(documentationLocation)
      ;

  // Short introduction.
  QString introduction = QObject::tr(
      "<p>"
      "%1 is one of the the user interfaces for the "
      "translational imaging platform called "
      "<a href=\"http://www.niftk.org\">%5</a>. "
      "%5 is co-developed by members of the <a href=\"%2\">%3 (%4) </a> at University College London (UCL)"
      " and <a href=\"https://www.kcl.ac.uk/lsm/research/divisions/imaging/about/People.aspx\">The School of Biomedical Engineering and Imaging Sciences</a>"
      " at King's College London (KCL)."
      "The Principal Investigator is <a href=\"https://kclpure.kcl.ac.uk/portal/sebastien.ourselin.html\">"
      "Prof. Sebastien Ourselin</a> at KCL "
      "and the team leader is <a href=\"https://iris.ucl.ac.uk/iris/browse/profile?upi=MJCLA42\">"
      "Dr Matt Clarkson</a> at UCL."
      "</p>"
      "<p>"
      "%5 was launched with funding from the NIHR and the Comprehensive Biomedical Research Centre "
      "at UCL and UCLH grant 168 and TSB grant M1638A. "
      "</p>"
      ).arg(applicationName).arg(originURL).arg(originLongText).arg(originShortText).arg(platformName);

  QString collaborators = QObject::tr(
      "<p>"
      "%1 is grateful for the continued support of our clinical and research collaborators including:"
      "<ul>"
      "<li>the <a href=\"http://dementia.ion.ucl.ac.uk/\">UCL Dementia Research Centre</a>.</li>"
      "<li>the <a href=\"http://www.ucl.ac.uk/ion/departments/neuroinflammation/\">UCL Department "
      "of Neuroinflammation</a>.</li>"
      "<li>the <a href=\"http://www.ucl.ac.uk/cabi/\">UCL Centre for Advanced Biomedical Imaging</a>.</li>"
      "<li>the <a href=\"http://www.ucl.ac.uk/surgery\">UCL Division Of Surgery And Interventional Science</a>.</li>"
      "</ul>"
      "In addition, the software development team would like to acknowledge the kind support "
      "of the open-source software community "
      "during development of NifTK and are especially grateful to the developers of "
      "<a href=\"http://www.mitk.org\">MITK</a> and <a href=\"http://www.commontk.org\">CTK</a>. "
      "In addition, various clip art comes from <a href=\"http://www.openclipart.org\">openclipart.org</a>."
      "</p>"
      ).arg(originShortText);

  // Over time, insert more software packages, as platform expands,
  // (and dependencies get exponentially more frustrating :-).
  QString versionsStart = QObject::tr(
      "<h3>Software Versions</h3>"
      "<p>"
      "%1 has been developed using the following core libraries."
      "</p>"
      "<p><table>"
      "<tr>"
      "<td><a href=\"http://www.boost.org\">Boost</a></td><td>%2</td>"
      "<td><a href=\"http://www.boost.org/LICENSE_1_0.txt\">Boost v1.0</a></td>"
      "<td><a href=\"%3\">from here</a></td>"
      "</tr>"
      "<tr>"
      "<td><a href=\"http://qt.nokia.com/products\">Qt</a></td><td>%4</td>"
      "<td><a href=\"http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html\">LGPL v2.1</a></td>"
      "<td><a href=\"http://qt.nokia.com/products\">from here</a></td>"
      "</tr>"
      "<tr>"
      "<td><a href=\"http://www.creatis.insa-lyon.fr/software/public/Gdcm/\">GDCM</a></td>"
      "<td>%5</td>"
      "<td><a href=\"http://www.creatis.insa-lyon.fr/software/public/Gdcm/License.html\">GDCM</a></td>"
      "<td><a href=\"%6\">from here</a></td>"
      "</tr>"
      "<tr>"
      "<td><a href=\"http://dicom.offis.de/\">DCMTK</a></td>"
      "<td>%7</td>"
      "<td><a href=\"ftp://dicom.offis.de/pub/dicom/offis/software/dcmtk/dcmtk360/COPYRIGHT\">DCMTK</a></td>"
      "<td><a href=\"%8\">from here</a></td>"
      "</tr>"
      "<tr>"
      "<td><a href=\"http://www.itk.org\">ITK</a>(Patched)</td>"
      "<td>%9</td>"
      "<td><a href=\"http://itk.org/ITK/project/license.html\">Apache v2.0</a></td>"
      "<td><a href=\"%10\">from here</a></td>"
      "</tr>"
      "<tr>"
      "<td><a href=\"http://www.vtk.org\">VTK</a></td>"
      "<td>%11</td>"
      "<td><a href=\"http://www.vtk.org/VTK/project/license.html\">BSD</a></td>"
      "<td><a href=\"%12\">from here</a></td>"
      "</tr>"
      "<tr>"
      "<td><a href=\"http://www.commontk.org\">CTK</a></td>"
      "<td>%13</td>"
      "<td><a href=\"http://www.apache.org/licenses/LICENSE-2.0.html\">Apache v2.0</a></td>"
      "<td><a href=\"%14\">from here</a></td>"
      "</tr>"
      "<tr>"
      "<td><a href=\"http://www.mitk.org\">MITK</a>(Modified)</td>"
      "<td>%15</td>"
      "<td><a href=\"http://www.mitk.org/wiki/License\">BSD-style</a></td>"
      "<td><a href=\"%16\">from here</a></td>"
      "</tr>"
      )
      .arg(applicationName)
      .arg(boostVersion)
      .arg(boostLocation)
      .arg(qtVersion)
      .arg(gdcmVersion)
      .arg(gdcmLocation)
      .arg(dcmtkVersion)
      .arg(dcmtkLocation)
      .arg(itkVersion)
      .arg(itkLocation)
      .arg(vtkVersion)
      .arg(vtkLocation)
      .arg(ctkVersion.left(10))
      .arg(ctkLocation)
      .arg(mitkVersion.left(10))
      .arg(mitkLocation)
      ;

  #ifdef USE_NIFTYREG
    QString niftyRegVersion(NIFTK_NIFTYREG_VERSION);
    QString niftyRegLocation(NIFTK_NIFTYREG_LOCATION);
    QString niftyRegText = QObject::tr(
        "<tr>"
        "<td><a href=\"http://sourceforge.net/projects/niftyreg/?source=directory\">NiftyReg</a></td>"
        "<td>%1</td>"
        "<td><a href=\"http://sourceforge.net/p/niftyreg/code/395/tree/trunk/nifty_reg/LICENSE.txt\">BSD</a></td>"
        "<td><a href=\"%2\">from here</a></td>"
        "</tr>"
        ).arg(niftyRegVersion).arg(niftyRegLocation);
  #endif

  #ifdef USE_NIFTYSEG
    QString niftySegVersion(NIFTK_NIFTYSEG_VERSION);
    QString niftySegLocation(NIFTK_NIFTYSEG_LOCATION);
    QString niftySegText = QObject::tr(
        "<tr>"
        "<td><a href=\"http://sourceforge.net/projects/niftyseg/?source=directory\">NiftySeg</a></td>"
        "<td>%1</td>"
        "<td><a href=\"http://sourceforge.net/p/niftyseg/code/145/tree/LICENSE.txt\">BSD</a></td>"
        "<td><a href=\"%2\">from here</a></td>"
        "</tr>"
        ).arg(niftySegVersion).arg(niftySegLocation);
  #endif

  #ifdef USE_NIFTYSIM
    QString niftySimVersion(NIFTK_NIFTYSIM_VERSION);
    QString niftySimLocation(NIFTK_NIFTYSIM_LOCATION);
    QString niftySimText = QObject::tr(
        "<tr>"
        "<td><a href=\"http://sourceforge.net/projects/niftysim/?source=directory\">NiftySim</a></td>"
        "<td>%1</td>"
        "<td><a href=\"http://sourceforge.net/p/niftysim/code/ci/master/tree/nifty_sim/LICENSE.txt\">BSD</a></td>"
        "<td><a href=\"%2\">from here</a></td>"
        "</tr>"
        ).arg(niftySimVersion).arg(niftySimLocation);
  #endif

  #ifdef BUILD_NiftyIGI
    QString niftyLinkVersion(NIFTK_NIFTYLINK_VERSION);
    //QString niftyLinkLocation(NIFTK_NIFTYLINK_LOCATION);
    QString niftyLinkText = QObject::tr(
      "<tr>"
       "<td><a href=\"https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyLink\">NiftyLink</a></td>"
       "<td>%1</td>"
       "<td><a href=\"https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyLink/blob/master/LICENSE.txt\">BSD</a></td>"
       "<td><a href=\"https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyLink\">from here</a></td>"
       "</tr>"
      ).arg(niftyLinkVersion.left(10));//.arg(niftyLinkLocation);

    QString arucoVersion(NIFTK_VERSION_ARUCO);
    QString arucoLocation(NIFTK_LOCATION_ARUCO);
    QString arucoText = QObject::tr(
      "<tr>"
      "<td><a href=\"http://www.uco.es/investiga/grupos/ava/node/26\">ARUCO</a></td>"
      "<td>%1</td>"
      "<td><a href=\"http://opensource.org/licenses/BSD-2-Clause\">BSD</a></td>"
      "<td><a href=\"%2\">from here</a></td>"
      "</tr>"
      ).arg(arucoVersion).arg(arucoLocation);

    QString eigenVersion(NIFTK_VERSION_EIGEN);
    QString eigenLocation(NIFTK_LOCATION_EIGEN);
    QString eigenText = QObject::tr(
      "<tr>"
      "<td><a href=\"http://eigen.tuxfamily.org/\">EIGEN</a></td>"
      "<td>%1</td>"
      "<td><a href=\"http://opensource.org/licenses/MPL-2.0\">MPL v2</a></td>"
      "<td><a href=\"%2\">from here</a></td>"
      "</tr>"
      ).arg(eigenVersion).arg(eigenLocation);
 
    QString aprilTagsVersion(NIFTK_VERSION_APRILTAGS);
    QString aprilTagsLocation(NIFTK_LOCATION_APRILTAGS);
    QString aprilTagsText = QObject::tr(
      "<tr>"
      "<td><a href=\"http://people.csail.mit.edu/kaess/apriltags/\">April Tags</a></td>"
      "<td>%1</td>"
      "<td><a href=\"http://opensource.org/licenses/LGPL-2.1\">LGPL v2.1</a></td>"
      "<td><a href=\"%2\">from here</a></td>"
      "</tr>"
      ).arg(aprilTagsVersion).arg(aprilTagsLocation);

    QString openCVVersion(NIFTK_VERSION_OPENCV);
    QString openCVLocation(NIFTK_LOCATION_OPENCV);
    QString openCVText = QObject::tr(
      "<tr>"
      "<td><a href=\"http://opencv.org/\">OpenCV</a></td>"
      "<td>%1</td>"
      "<td><a href=\"https://github.com/Itseez/opencv/blob/master/doc/license.txt\">BSD</a></td>"
      "<td><a href=\"%2\">from here</a></td>"
      "</tr>"
      ).arg(openCVVersion).arg(openCVLocation);
    
  #endif

  #ifdef BUILD_PCL
    QString flannVersion(NIFTK_VERSION_FLANN);
    QString flannLocation(NIFTK_LOCATION_FLANN);
    QString flannText = QObject::tr(
      "<tr>"
      "<td><a href=\"http://www.cs.ubc.ca/research/flann/\">FLANN</a>(Patched)</td>"
      "<td>%1</td>"
      "<td><a href=\"http://opensource.org/licenses/BSD-3-Clause\">BSD</a></td>"
      "<td><a href=\"%2\">from here</a></td>"
      "</tr>"
      ).arg(flannVersion).arg(flannLocation);

    QString pclVersion(NIFTK_VERSION_PCL);
    QString pclLocation(NIFTK_LOCATION_PCL);
    QString pclText = QObject::tr(
      "<tr>"
      "<td><a href=\"http://pointclouds.org/\">PCL</a></td>"
      "<td>%1</td>"
      "<td><a href=\"https://github.com/PointCloudLibrary/pcl/blob/master/LICENSE.txt\">BSD</a></td>"
      "<td><a href=\"%2\">from here</a></td>"
      "</tr>"
      ).arg(pclVersion).arg(pclLocation);
  #endif

  #ifdef BUILD_Caffe
    QString glogVersion(NIFTK_VERSION_glog);
    QString glogLocation(NIFTK_LOCATION_glog);
    QString glogText = QObject::tr(
      "<tr>"
      "<td><a href=\"https://github.com/google/glog\">glog</a></td>"
      "<td>%1</td>"
      "<td><a href=\"https://github.com/google/glog/blob/master/COPYING\">BSD</a></td>"
      "<td><a href=\"%2\">from here</a></td>"
      "</tr>"
      ).arg(glogVersion).arg(glogLocation);
    
    QString gflagsVersion(NIFTK_VERSION_gflags);
    QString gflagsLocation(NIFTK_LOCATION_gflags);
    QString gflagsText = QObject::tr(
      "<tr>"
      "<td><a href=\"https://github.com/gflags/gflags/\">gflags</a></td>"
      "<td>%1</td>"
      "<td><a href=\"https://github.com/gflags/gflags/blob/master/COPYING.txt\">BSD</a></td>"
      "<td><a href=\"%2\">from here</a></td>"
      "</tr>"
      ).arg(gflagsVersion).arg(gflagsLocation);

    QString hdf5Version(NIFTK_VERSION_HDF5);
    QString hdf5Location(NIFTK_LOCATION_HDF5);
    QString hdf5Text = QObject::tr(
      "<tr>"
      "<td><a href=\"https://support.hdfgroup.org/HDF5/\">HDF5</a></td>"
      "<td>%1</td>"
      "<td><a href=\"https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.0/src/unpacked/COPYING\">"
      "BSD</a></td>"
      "<td><a href=\"%2\">from here</a></td>"
      "</tr>"
      ).arg(hdf5Version).arg(hdf5Location);

    QString protoBufVersion(NIFTK_VERSION_ProtoBuf);
    QString protoBufLocation(NIFTK_LOCATION_ProtoBuf);
    QString protoBufText = QObject::tr(
      "<tr>"
      "<td><a href=\"https://github.com/google/protobuf\">ProtoBuf</a></td>"
      "<td>%1</td>"
      "<td><a href=\"https://github.com/google/protobuf/blob/master/LICENSE\">BSD</a></td>"
      "<td><a href=\"%2\">from here</a></td>"
      "</tr>"
      ).arg(protoBufVersion).arg(protoBufLocation);

    QString openBLASBufVersion(NIFTK_VERSION_OpenBLAS);
    QString openBLASLocation(NIFTK_LOCATION_OpenBLAS);
    QString openBLASBufText = QObject::tr(
      "<tr>"
      "<td><a href=\"http://www.openblas.net/\">OpenBLAS</a></td>"
      "<td>%1</td>"
      "<td><a href=\"https://github.com/xianyi/OpenBLAS/blob/develop/LICENSE\">BSD</a></td>"
      "<td><a href=\"%2\">from here</a></td>"
      "</tr>"
      ).arg(openBLASBufVersion).arg(openBLASLocation);

    QString caffeVersion(NIFTK_VERSION_Caffe);
    QString caffeLocation(NIFTK_LOCATION_Caffe);
    QString caffeText = QObject::tr(
      "<tr>"
      "<td><a href=\"http://caffe.berkeleyvision.org/\">Caffe</a></td>"
      "<td>%1</td>"
      "<td><a href=\"https://github.com/BVLC/caffe/blob/master/LICENSE\">BSD</a></td>"
      "<td><a href=\"%2\">from here</a></td>"
      "</tr>"
      ).arg(caffeVersion).arg(caffeLocation);

  #endif

  QString versionsEnd = QObject::tr(
      "</table></p>"
      );

  QString furtherInformation = QObject::tr(
      "<p>"
      "Further information can be obtained by:"
      "<ul>"
      "<li>Emailing the %1 <a href=\"%2\">users mailing list</a>.</li>"
      "</ul>"
      "</p>"
      ).arg(platformName).arg(userContact);

  // Stick it all together.
  QString totalText =
      titles
      .append(introduction)
      .append(collaborators)
      .append(furtherInformation)
      .append(versionsStart)
#ifdef USE_NIFTYREG
      .append(niftyRegText)
#endif
#ifdef USE_NIFTYSEG
      .append(niftySegText)
#endif
#ifdef USE_NIFTYSIM
      .append(niftySimText)
#endif
#ifdef BUILD_NiftyIGI
      .append(niftyLinkText)
      .append(arucoText)
      .append(eigenText)
      .append(aprilTagsText)
      .append(openCVText)
#endif
#ifdef BUILD_PCL
      .append(flannText)
      .append(pclText)
#endif
#ifdef BUILD_Caffe
      .append(glogText)
      .append(gflagsText)
      .append(hdf5Text)
      .append(protoBufText)
#ifndef __APPLE__
      .append(openBLASBufText)
#endif
      .append(caffeText)
#endif
      .append(versionsEnd)
      ;

  this->setWindowTitle(tr("About %1").arg(applicationName));
  QIcon helpAboutIcon(QLatin1String(":/Icons/icon.png"));

  if (!helpAboutIcon.isNull())
  {
    this->setWindowIcon(helpAboutIcon);
  }

  m_HelpAboutLabel->setText(totalText);

}

} // end namespace
