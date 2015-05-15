/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#ifndef PointSetConverterView_h
#define PointSetConverterView_h

#include <berryISelectionListener.h>

#include <QmitkAbstractView.h>

#include <mitkPlanarCircle.h>

#include "ui_PointSetConverterViewControls.h"


/**
  \brief PointSetConverterView

  \warning  This class is not yet documented. Use "git blame" and ask the author to provide basic documentation.

  \sa QmitkAbstractView
  \ingroup ${plugin_target}_internal
*/
class PointSetConverterView : public QmitkAbstractView
{
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT

  public:

    PointSetConverterView();

    virtual ~PointSetConverterView();

    static const std::string VIEW_ID;

  protected slots:

    /// \brief Create a new empty point set in the data manager
    void OnCreateNewPointSetButtonClicked();

    /// \brief Convert polygons in the datamanager to point sets with similar names
    void OnConvertPolygonsToPointSetButtonClicked();

  protected:

    virtual void CreateQtPartControl(QWidget *parent);

    virtual void SetFocus();

    /// \brief called by QmitkFunctionality when DataManager's selection has changed
    /// If the selected node is a point set -- it will be changed to the active point set
    virtual void OnSelectionChanged( berry::IWorkbenchPart::Pointer source,
                                     const QList<mitk::DataNode::Pointer>& nodes );

    Ui::PointSetConverterViewControls* m_Controls;

  private:

    /// \brief Caclulate the centroid of a mitkPlanarCircle 
    mitk::Point3D PlanarCircleToPoint( const mitk::PlanarCircle* circle);
    
    /// \breif Find the Point set with the given name
    mitk::PointSet::Pointer FindPointSetNode( const std::string& name );

    mitk::DataNode::Pointer m_ReferenceImage;
    QWidget* m_Parent;

};

#endif // PointSetConverterView_h
