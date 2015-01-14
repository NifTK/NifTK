"""
   Interface for niftk filter tools
"""

from niftk.base import NIFTKCommand, NIFTKCommandInputSpec, getNiftkPath
from nipype.interfaces.base import (TraitedSpec, File, traits)


import os.path
import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt

from nipype.utils.filemanip import split_filename


class InterSliceCorrelationPlotInputSpec(NIFTKCommandInputSpec):    
    in_file = File(argstr="%s", exists=True, mandatory=True,
                        desc="Input image")
    bval_file = File(argstr="%s", exists=True, mandatory=False,
                        desc="Input bval file")
    
class InterSliceCorrelationPlotOutputSpec(TraitedSpec):
    out_file   = File(exists=False, genfile = True,
                      desc="Interslice correlation plot")

class InterSliceCorrelationPlot(NIFTKCommand):
    input_spec = InterSliceCorrelationPlotInputSpec
    output_spec = InterSliceCorrelationPlotOutputSpec
    _suffix = "_interslice_ncc"
    
    def _gen_output_filename(self, in_file):
        _, base, _ = split_filename(in_file)
        outfile = base + self._suffix + '.pdf'
        return os.path.abspath(outfile)
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.out_file
        return outputs
    
    def _run_interface(self, runtime):
        # Load the original image
        nib_image = nib.load(self.inputs.in_file)
        data = nib_image.get_data()
        dim = data.shape
        vol_number = 1;
        if len(dim)>3:
            vol_number=dim[3]
        if self.inputs.bval_file:
            bvalues=np.loadtxt(self.inputs.bval_file)
            if len(bvalues) != vol_number:
                exit
        cc_values = np.zeros((vol_number,len(range(1,dim[2]-1))))
        b0_values = np.zeros((vol_number,len(range(1,dim[2]-1))))
        for v in range(vol_number):
            if vol_number>1:
                volume=data[:,:,:,v]
            else:
                volume=data[:,:,:]
            current_b0=False
            if self.inputs.bval_file:
                if bvalues[v]<20:
                    current_b0=True
            temp_array1=volume[:,:,0];
            voxel_number = temp_array1.size
            temp_array1=np.reshape(temp_array1,voxel_number)
            temp_array1 = (temp_array1 - np.mean(temp_array1)) / np.std(temp_array1)
            for z in range(1,dim[2]-1):
                temp_array2=np.reshape(volume[:,:,z],voxel_number)
                temp_array2 = (temp_array2 - np.mean(temp_array2)) / np.std(temp_array2)
                if current_b0==True:
                    cc_values[v,z-1]=np.nan
                    b0_values[v,z-1]=(np.correlate(temp_array1,temp_array2) / np.double(voxel_number))
                else:
                    b0_values[v,z-1]=np.nan
                    cc_values[v,z-1]=(np.correlate(temp_array1,temp_array2) / np.double(voxel_number))
                temp_array1=np.copy(temp_array2)
        fig = plt.figure(figsize = (vol_number/4,6))
        mask_cc_values = np.ma.masked_array(cc_values,np.isnan(cc_values))
        mask_b0_values = np.ma.masked_array(b0_values,np.isnan(b0_values))
        mean_cc_values=np.mean(mask_cc_values[:,:],axis=0)
        mean_b0_values=np.mean(mask_b0_values[:,:],axis=0)
        std_cc_values=np.std(mask_cc_values[:,:],axis=0)
        std_b0_values=np.std(mask_b0_values[:,:],axis=0)
        x_axis = np.array(range(1,dim[2]-1))
        mpl.rcParams['text.latex.unicode']=True
        plt.plot(x_axis , mean_cc_values , 'b-',
                         label='Mean non B0 $(\pm 3.5 \sigma)$')
        plt.plot(x_axis , mean_b0_values , 'r-',
                         label='Mean B0 $(\pm 3.5 \sigma)$')
        sigma_mul=3.5
        plt.fill_between(x_axis,
                         mean_cc_values - sigma_mul*std_cc_values,
                         mean_cc_values + sigma_mul*std_cc_values,
                         facecolor='b',
                         linestyle='dashed',
                         alpha=0.2)
        plt.fill_between(x_axis,
                         mean_b0_values - sigma_mul*std_b0_values,
                         mean_b0_values + sigma_mul*std_b0_values,
                         facecolor='r',
                         linestyle='dashed',
                         alpha=0.2)       
        for v in range(vol_number):
            current_b0=False
            current_label=False
            if self.inputs.bval_file:
                if bvalues[v]<20:
                    current_b0=True
            if current_b0==True:
                for z in range(1,dim[2]-1):
                    if b0_values[v,z-1] < mean_b0_values[z-1]-sigma_mul*std_b0_values[z-1]:
                        if current_label:
                            plt.plot(z, b0_values[v,z-1], '.',
                                     color=(plt.cm.jet(v*255/vol_number)[0:3]))
                        else:
                            current_label=True
                            plt.plot(z, b0_values[v,z-1], '.',
                                     color=(plt.cm.jet(v*255/vol_number)[0:3]),
                                     label='Volume '+str(v))
            else:
                for z in range(1,dim[2]-1):
                    if cc_values[v,z-1] < mean_cc_values[z-1]-sigma_mul*std_cc_values[z-1]:
                        if current_label:
                            plt.plot(z, cc_values[v,z-1], '.',
                                     color=(plt.cm.jet(v*255/vol_number)[0:3]))
                        else:
                            current_label=True
                            plt.plot(z, cc_values[v,z-1], '.',
                                     color=(plt.cm.jet(v*255/vol_number)[0:3]),
                                     label='Volume '+str(v))
        plt.ylabel('Normalised cross-corelation')
        plt.xlabel('Slice')
        plt.title('Inter-slice normalised cross-correlation\n' + \
                  'Scan: '+os.path.basename(self.inputs.in_file))
        plt.xticks(np.arange(0, dim[2], 2.0))
        plt.ylim([0,1])
        plt.legend(loc='best', numpoints=1, fontsize='small')
        self.out_file = self._gen_output_filename(self.inputs.in_file)
        fig.savefig(self.out_file, format='PDF')
        plt.close()
        return runtime
        




class MatrixRotationPlotInputSpec(NIFTKCommandInputSpec):    
    in_files = traits.List(File(exists=True),
                           exists=True,
                           mandatory=True,
                           desc="List of input transformation matrix files")
    
class MatrixRotationPlotOutputSpec(TraitedSpec):
    out_file   = File(exists=False, genfile = True,
                      desc="Matrix rotation plot")

class MatrixRotationPlot(NIFTKCommand):
    input_spec = MatrixRotationPlotInputSpec
    output_spec = MatrixRotationPlotOutputSpec
    _suffix = "_rotation"
    
    def _gen_output_filename(self, in_file):
        _, base, _ = split_filename(in_file)
        outfile = base + self._suffix + '.pdf'
        return os.path.abspath(outfile)
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.out_file
        return outputs
        
    def get_max_norm_row(self, in_matrix):
       return np.max([
           np.sum(np.fabs(in_matrix[:][0])),
           np.sum(np.fabs(in_matrix[:][1])),
           np.sum(np.fabs(in_matrix[:][2]))
           ])
        
    def get_max_norm_col(self, in_matrix):
       return np.max([
           np.sum(np.fabs(in_matrix[0][:])),
           np.sum(np.fabs(in_matrix[1][:])),
           np.sum(np.fabs(in_matrix[2][:]))
           ])
       
    def polar_decomposition(self, in_matrix):
        gam = np.linalg.det(in_matrix)
        while gam == 0.0:
            gam = 0.00001 * ( 0.001 + self.get_max_norm_row(in_matrix) ) ;
            in_matrix[0][0] += gam
            in_matrix[1][1] += gam
            in_matrix[2][2] += gam ;
            gam = np.linalg.det(in_matrix)
        dif=1
        k=0
        while True:
            matrix_inv = np.linalg.inv(in_matrix)
            if dif > 0.3:
                alp = np.sqrt(self.get_max_norm_row(in_matrix) * \
                              self.get_max_norm_col(in_matrix) )
                bet = np.sqrt(self.get_max_norm_row(matrix_inv) * \
                              self.get_max_norm_col(matrix_inv) ) ;
                gam = np.sqrt( bet / alp ) ;
                gmi = 1 / gam
            else:
                gam = 1.0
                gmi = 1.0
            
            temp_matrix = 0.5 * (gam*in_matrix + gmi*np.transpose(matrix_inv))
            
            dif =np.fabs(temp_matrix[0][0]-in_matrix[0][0])+np.fabs(temp_matrix[0][1]-in_matrix[0][1]) \
                +np.fabs(temp_matrix[0][2]-in_matrix[0][2])+np.fabs(temp_matrix[1][0]-in_matrix[1][0]) \
                +np.fabs(temp_matrix[1][1]-in_matrix[1][1])+np.fabs(temp_matrix[1][2]-in_matrix[1][2]) \
                +np.fabs(temp_matrix[2][0]-in_matrix[2][0])+np.fabs(temp_matrix[2][1]-in_matrix[2][1]) \
                +np.fabs(temp_matrix[2][2]-in_matrix[2][2])
            k = k+1
            if k > 100 or dif < 3.e-6:
                break
        return temp_matrix

    def extract_quaternion(self, in_matrix):
        xd = np.sqrt(np.sum(np.square(np.double(in_matrix[:][0]))))
        yd = np.sqrt(np.sum(np.square(np.double(in_matrix[:][1]))))
        zd = np.sqrt(np.sum(np.square(np.double(in_matrix[:][2]))))
        if xd == 0:
            in_matrix[0][0]=1
            in_matrix[1][0]=0
            in_matrix[2][0]=0
            xd=1
        else: 
            in_matrix[:][0]/=xd
        if yd == 0:
            in_matrix[0][1]=0
            in_matrix[1][1]=1
            in_matrix[2][1]=0
            yd=1
        else:
            in_matrix[:][1]/=yd
        if zd == 0:
            in_matrix[0][2]=0
            in_matrix[1][2]=0
            in_matrix[2][2]=1
            zd=1
        else:
            in_matrix[:][2]/=zd

        temp_matrix=self.polar_decomposition(in_matrix)
        det=np.linalg.det(temp_matrix)
        if det>0:
            qfac=1.0
        else:
            qfac=-1
            temp_matrix[0][2]=-temp_matrix[0][2]
            temp_matrix[1][2]=-temp_matrix[1][2]
            temp_matrix[2][2]=-temp_matrix[2][2]

        a = temp_matrix[0][0] + temp_matrix[1][1] + temp_matrix[2][2] + 1

        if a > 0.5:
            a = 0.5  * np.sqrt(a)
            b = 0.25 * (temp_matrix[2][1]-temp_matrix[1][2]) / a
            c = 0.25 * (temp_matrix[0][2]-temp_matrix[2][0]) / a
            d = 0.25 * (temp_matrix[1][0]-temp_matrix[0][1]) / a
        else:
            xd = 1 + temp_matrix[0][0] - (temp_matrix[1][1]+temp_matrix[2][2])
            yd = 1 + temp_matrix[1][1] - (temp_matrix[0][0]+temp_matrix[2][2])
            zd = 1 + temp_matrix[2][2] - (temp_matrix[0][0]+temp_matrix[1][1])
            if xd > 1.0:
                b = 0.5  * np.sqrt(xd)
                c = 0.25 * (temp_matrix[0][1]+temp_matrix[1][0]) / b
                d = 0.25 * (temp_matrix[0][2]+temp_matrix[2][0]) / b
                a = 0.25 * (temp_matrix[2][1]-temp_matrix[1][2]) / b
            elif yd > 1.0:
                c = 0.5  * np.sqrt(yd) ;
                b = 0.25 * (temp_matrix[0][1]+temp_matrix[1][0]) / c
                d = 0.25 * (temp_matrix[1][2]+temp_matrix[2][1]) / c
                a = 0.25 * (temp_matrix[0][2]-temp_matrix[2][0]) / c
            else:
                d = 0.50 * np.sqrt(zd) ;
                b = 0.25 * (temp_matrix[0][2]+temp_matrix[2][0]) / d
                c = 0.25 * (temp_matrix[1][2]+temp_matrix[2][1]) / d
                a = 0.25 * (temp_matrix[1][0]-temp_matrix[0][1]) / d
        if a < 0:
            b=-b
            c=-c
            d=-d
            a=-a
        out_values={'a':a,'b':b,'c':c,'d':d,'qfac':qfac}
        return out_values
    
    def _run_interface(self, runtime):
        all_matrices = self.inputs.in_files
        num_matrices = len(all_matrices)
        rotation_x = np.zeros(num_matrices)
        rotation_y = np.zeros(num_matrices)
        rotation_z = np.zeros(num_matrices)
        for i in range(0,num_matrices):
            matrix = np.loadtxt(all_matrices[i])
            values = self.extract_quaternion(matrix[np.ix_([0,1,2],[0,1,2])])
            rotation_x[i]=np.arctan(2*(values['a']*values['b']+values['c']*values['d'])/(1-2*(np.square(values['b'])+np.square(values['c']))))
            rotation_y[i]=np.arcsin(2*(values['a']*values['c'] - values['b']*values['d']))
            rotation_z[i]=np.arctan(2*(values['a']*values['d']+values['b']*values['c'])/(1-2*(np.square(values['c'])+np.square(values['d']))))
        rotation_x = rotation_x - np.mean(rotation_x)
        rotation_y = rotation_y - np.mean(rotation_y)
        rotation_z = rotation_z - np.mean(rotation_z)
        fig = plt.figure()
        x_axis=np.arange(0,num_matrices)
        plt.plot(x_axis,np.rad2deg(rotation_x),'r-', label='x-axis rotation')
        plt.plot(x_axis,np.rad2deg(rotation_y),'b-', label='y-axis rotation')
        plt.plot(x_axis,np.rad2deg(rotation_z),'k-', label='z-axis rotation')
        plt.ylabel('Demeaned rotation value (in degrees)')
        plt.xlabel('Volume')
        plt.title('Recovered rotation per volume in degrees')
        plt.legend(loc='best', fontsize='small')
        plt.ylim([-15,15])
        self.out_file = self._gen_output_filename(all_matrices[0])
        fig.savefig(self.out_file, format='PDF')
        plt.show()
#        plt.close()
        return runtime
        
