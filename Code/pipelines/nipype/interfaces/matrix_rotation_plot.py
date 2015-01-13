#!/usr/bin/env python

import os.path
import numpy as np
import matplotlib.pyplot as plt

from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces.base import (TraitedSpec, File, traits)


class MatrixRotationPlotInputSpec(BaseInterfaceInputSpec):    
    in_files = traits.List(File(exists=True),
                           exists=True,
                           mandatory=True,
                           desc="List of input transformation matrix files")
    
class MatrixRotationPlotOutputSpec(TraitedSpec):
    out_file   = File(exists=False, genfile = True,
                      desc="Matrix rotation plot")

class MatrixRotationPlot(BaseInterface):
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
        