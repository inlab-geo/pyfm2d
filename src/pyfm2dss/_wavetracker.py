import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ._pyfm2dss import FastMarchingMethod
from . import _bases as base
from scipy.interpolate import RectBivariateSpline

# The idea is to create high level classes to facilitate in inversion. Possible clases are
#
# - Survey to hold observations and predictions
# - basisModel to define the velocity model in different basis classes
# - WaveTracker to call FMM
# - Plotter plotting functions...
#--------------------------------------------------------------------------------------------

# This library is a python interface to Nick Rawlinson's 2D Fast Marching Fortran package fm2dss.f90
#
# It uses the ctypes interface to fm2dss.f90 developed by Juerg Hauser in file "._pyfm2dss"
#
# History:
#        January 2025:  Uses ctypes interface in pyfm2dss.py 
#        January 2024:  Updated to interface with package bases.py allowing multiple 2D model bases 
#                       including pixel and discrete cosine. Also interfaces with Jacobian integration 
#                       package jip.py to calculate Frechect kernels when model bases 
#                       are not simple pixel bases, e.g discrete cosine bases.
#
#        Definitions within waveTracker follow conventions from Andrew Valentine's rayTracer.py package.
#
# M. Sambridge 
# January 2025
#--------------------------------------------------------------------------------------------
 
class WaveTracker:
    
    def __init__(self):
        self.fmm=FastMarchingMethod()
    
    def setTimes(self,t):
        self.ttimes = t.copy()
        
    def setPaths(self,p):
        self.paths = p.copy()
        
    def setFrechet(self,f):
        self.frechet = f.copy()
        
    def setTfield(self,w,source):
        self.tfield = w.copy()
        self.tfieldsource = source
        
    def calc_wavefronts(self,v,recs,srcs,
                        verbose=False,paths=False,frechet=False,times=True,tfieldsource=-1,
                        sourcegridrefine=True,sourcedicelevel=5,sourcegridsize=10,
                        earthradius=6371.0,schemeorder=1,nbsize=0.5,degrees=False,velocityderiv=False,
                        extent=[0.,1.,0.,1.],dicex=8,dicey=8):
        
        '''

        A function to perform 2D Fast Marching of wavefronts from sources in a 2D velocity model.

        Inputs:
            v, ndarray(nx,ny)          : coefficients of velocity field in 2D grid
            recs, ndarray(nr,2)        : receiver locations (x,y)
            srcs, ndarray(ns,2)        : source locations (x,y)
            paths, bool                : raypath option (True=calculate ad return ray paths)
            frechet, bool              : frechet derivative option (True=calculate and return frechet derivative matrix for raypths in each cell)
            times, bool                : travel times derivative option (True=calculate and travel times)
            tfieldsource, int          : source id to calculate travel time field (0=none,>0=source id)
            sourcegridrefine, bool     : Apply sourcegrid refinement (default=True)
            sourcedicelevel, int       : Source discretization level. Number of sub-divisions per cell (default=5, i.e. 1 model cell becomes 5x5 sub-cells)
            sourcegridsize, int        : Number of model cells to refine about source at sourcedicelevel (default=10, i.e. 10x10 cells are refines about source)
            earthradius, float         : radius of Earth in km, used for spherical to Cartesian transform (default=6371.0)
            schemeorder, int           : switch to use first order (0) or mixed order(1) scheme (default=1,mixed)
            nbsize,float               : Narrow band size (0-1) as fraction of nnx*nnz (default=0.5)
            degrees, bool              : True if input distances are in degrees (default=False). Uses earthradius to convert to km.
            velocityderiv, bool        : Switch to return Frechet derivatives of travel times w.r.t. velocities (True) rather than slownesses (False, default).
            extent, list               : 4-tuple of model extent [xmin,xmax,ymin,ymax]. (default=[0.,1.,0.,1.])
            dicex, int                 : x-subgrid discretization level for B-spline interpolation of input model (default=8)
            dicey, int                 : y-subgrid discretization level for B-spline interpolation of input model (default=8)
                
                                    
            For a complete explanation of input parameters see Rawlinson's instruction manual for fmtomo package
            (https://iearth.edu.au/codes/FMTOMO/instructions.pdf)
        '''

        recs = recs.reshape(-1, 2)
        srcs = srcs.reshape(-1, 2)
        if(tfieldsource+1 > len(srcs)): # source requested for travel time field does not exist
            print('Error: Travel time field corresponds to source:',tfieldsource,'\n',
                  '      but total number of sources is',len(srcs),
                  '\n       No travel time field will be calculated.\n')   
        
        # fmst expects input spatial co-ordinates in degrees and velocities in kms/s so we adjust (unless degrees=True)

        kms2deg = 180./(earthradius*np.pi)

        #write out input files for Fast Marching wavefront tracker fmst

        #write_fm2dss_input(wdir,paths=paths,frechet=frechet,times=times,tfieldsource=tfieldsource+1,
        #                   dicex=dicex,dicey=dicey,sourcegrid=sourcegrid,sourcedice=sourcedice,
        #                   sourcegridsize=sourcegridsize,earthradius=earthradius,schemeorder=schemeorder,nbsize=nbsize,
        #                   tfilename=tfilename,ffilename=ffilename,wfilename=wfilename,rfilename=rfilename)  # write out control file
    
        lpaths = 0                    # Write out raypaths (<0=all,0=no,>0=source id)
        if(paths): lpaths=-1          # only allow all or none
    
        lttimes=0                     # int to calculate travel times (y=1,n=0)
        if(times): lttimes = 1
    
        lfrechet=0                    # bool to calculate Frechet derivatives of travel times w.r.t. slownesses (0=no,1=yes)
        if(frechet): lfrechet = 1     
        
        self.fmm.set_solver_options(dicex,dicey,
                                    int(sourcegridrefine),
                                    sourcedicelevel,
                                    sourcegridsize,
                                    earthradius,
                                    schemeorder,
                                    nbsize,
                                    lttimes,
                                    lfrechet,
                                    tfieldsource+1,
                                    lpaths)


        #write_rs(recs,srcs,wdir)    # write out sources and receiver files
        
        self.fmm.set_sources(srcs[:,1],srcs[:,0]     # set sources

        self.fmm.set_receivers(recs[:,1],recs[:,0])  # set receivers

        #self.fmm.set_velocity_model(nvx, nvz, goxd, gozd, dvxd, dvzd, velv)


        noncushion,nodemap = write_gridc(v,extent,wdir) # write data for input velocity model file gridc.txc
    
        write_otimes([[True]*len(recs)]*len(srcs),wdir) # write out rays to be calculated
    
        # run fmst wavefront tracker code from command line
    
        command = "cd "+wdir+"; ./fm2dss"
        out = subprocess.run(command,stdout=subprocess.PIPE, text=True,shell=True)
        if(verbose): print(' Message from fmm2dss:',out.stdout)
        if(out.returncode != 0):
            print(' The process returned with errorcode:',out.returncode)
            print(' stdout: \n',out.stdout)
            print(' stderr: \n',out.stderr)
            return
        
        # collect results
        if(times):
            ttimes = read_fmst_ttimes(wdir+'/'+tfilename)
            if(not degrees): ttimes*= kms2deg # adjust travel times because inputs are not in degrees
    
        if(paths): 
            raypaths = read_fmst_raypaths(wdir+'/'+rfilename)
    
        if(frechet):
            frechetvals = read_fmst_frechet(wdir+'/'+ffilename,noncushion,nodemap)
            if(not degrees): frechetvals*= kms2deg # adjust travel times because inputs are not in degrees
            if(not velocityderiv): 
                x2 = -(v*v).reshape(-1)
                frechetvals = frechetvals.multiply(x2)

        if(tfieldsource>=0):
            tfieldvals = read_fmst_wave(wdir+'/'+wfilename)
            if(not degrees): tfieldvals*= kms2deg # adjust travel times because inputs are not in degrees
        
        #   build class object to return
        result = fmmResult()
    
        if(times): result.setTimes(ttimes)
        if(paths): result.setPaths(raypaths)
        if(frechet): result.setFrechet(frechetvals)
        if(tfieldsource >-1): result.setTfield(tfieldvals,tfieldsource) # set traveltime field and source id
        return result

		
class gridModel(object): # This is for the original regular model grid (without using the basis.py package)
    
    def __init__(self,velocities,extent=(0,1,0,1)):
        self.nx,self.ny = velocities.shape
        self.velocities=velocities
        self.xmin,self.xmax,self.ymin,self.ymax = extent
        self.xx = np.linspace(self.xmin,self.xmax,self.nx+1)
        self.yy = np.linspace(self.ymin,self.ymax,self.ny+1)
        #self.dicex = dicex
        #self.dicey = dicey
        self.extent = extent
    def getVelocity(self):
        return self.velocities.copy()
    def getSlowness(self):
        return 1./self.velocities # No copy needed as operation must return copy
    def setVelocity(self,v):
        assert self.velocities.shape == v.shape
        self.velocities = v.copy()
    def setSlowness(self,s):
        assert self.velocities.shape == s.shape
        self.velocities = 1./s
        
class basisModel(object): # This is for a 2D model basis accessed through the package basis.py
    '''

    A model class which is an interface to package basis.py to incorporate local or global 2D bases for tomography.

    Handles cases where model bases are local 2D pixels (default) or global 2D functions, e.g. cosine basis. 
    Handles cases where input coefficients are velocities (default) or slownesses.

    Inputs:
        coeffs, ndarray(nx,ny)          : coefficients of velocity or slowness field in selected basis
        coeff_type, string              : ='velocities' then coefficients are velocities; 'slownesses' for slowness coefficients
        basis, string                   : type of model basis function 
                                          `2Dpixel' for 2D regular grid of velocity/slowness values; 
                                          '2Dcosine' for 2D cosine basis functions.
        ref, float                      : reference value used for perturbative representation, 
                                          i.e. v[x,y] = ref + coeff[i]*basis[i,x,y]; or s[x,y] = ref + coeff[i]*basis[i,x,y], (i=1,...nx*ny)
                                    
    '''

    def __init__(self,coeffs,extent=(0,1,0,1),ref=0.0,coeff_type='velocities',basis='2Dpixel'):
        self.nx,self.ny = coeffs.shape
        self.coeffs=coeffs
        self.xmin,self.xmax,self.ymin,self.ymax = extent
        self.extent = extent
        self.dx = (self.xmax - self.xmin)/self.nx
        self.dy = (self.ymax - self.ymin)/self.ny
        self.basis_type = basis
        self.A_calc = False
        self.vref,self.sref = 0., 0. # reference values
        if(ref != 0.):
            if(coeff_type =='velocities'):
                self.vref,self.sref = ref,1./ref
            else:
                self.vref,self.sref = 1./ref,ref
        if(self.basis_type == '2Dpixel'):
            self.xx = np.linspace(self.xmin,self.xmax,self.nx+1)
            self.yy = np.linspace(self.ymin,self.ymax,self.ny+1)
            #self.dicex = dicex
            #self.dicey = dicey
            self.basis = base.pixelbasis2D(self.xx,self.yy)
        elif(self.basis_type == '2Dcosine'):
            self.basis = base.cosinebasis2D(self.xmin,self.xmax,self.ymin,self.ymax,self.nx,self.ny,npoints=[120,120,200])
        
        self.coeff_type = coeff_type # need to know this for non-pixel bases

    def getVelocity(self,nx=None,ny=None,returncoeff=False): 
        ''' With no arguments this will return a velocity field. 
            If bases are 2D pixels the keyword returncoeff is ignored
            If bases are not pixels and returncoeff is False, then a velocity field is evaluated and returned.
            If bases are not 2D pixels and returncoeff is True, then velocity basis coefficients are returned.
            Default values of (nx,ny) are determined by .getImage() and are the input resolution of the model.
        '''   
        if(self.basis_type == '2Dpixel'):
            if(self.coeff_type=='velocities'):
                if(returncoeff): return self.coeffs.copy()  # because coefficients must be for velocities
                return self.vref + self.coeffs.copy()        # because coefficients must be for velocities
            else:
                if(returncoeff): return 1./self.coeffs      # because coefficients must be for slownesses
                return 1./(self.sref + self.coeffs)          # because coefficients must be for slownesses
        else:                                  # non pixel basis
            if(self.coeff_type=='velocities'):
                if(returncoeff): return self.coeffs.copy()  # coefficients are velocities and so we return coefficients
                return self.vref + self.getImage(nx=nx,ny=ny) # return a velocity field evaluated from basis summation
            else:                               
                if(returncoeff): return self.fitcoefficientsS2V() # coefficients are slownesses and we need to find equivalent velocity coefficients
                return 1./(self.sref+self.getImage(nx=nx,ny=ny)) # coefficients are slownesses and we must return a velocity field

    def basis_transform_matrix(self):
        if(not self.A_calc):
            A = np.zeros((self.basis.nbases,self.nx*self.ny))
            for j in range(self.basis.nbases):
                A[j] = self.getbasisImage(j,nx=self.nx,ny=self.ny).flatten()
            self.A = A
            self.A_calc = True
        return self.A
        
    def fitcoefficientsV2S(self,nx=None,ny=None): # calculate slowness coefficients that correspond to a given set of velocity coefficients in model basis
        if(nx==None): nx = self.nx
        if(ny==None): ny = self.ny
        vtarget = self.getVelocity(nx=nx,ny=ny)-self.vref
        if(not self.A_calc): 
            A = self.basis_transform_matrix()
            self.A = A
        slowcoeff,res,rank,s = np.linalg.lstsq(self.A.T, 1./vtarget.flatten(),rcond=None) # fit slownesses coefficients to slowness field
        return slowcoeff.reshape(self.nx,self.ny)
    
    def fitcoefficientsS2V(self,nx=None,ny=None): # calculate velocity coefficients that correspond to a given set of slowness coefficients in model basis
        if(nx==None): nx = self.nx
        if(ny==None): ny = self.ny
        starget = self.getSlowness(nx=nx,ny=ny)-self.sref # get slowness field perturbation
        if(not self.A_calc): 
            A = self.basis_transform_matrix()
            self.A = A
        velcoeff,res,rank,s = np.linalg.lstsq(self.A.T, 1./starget.flatten(),rcond=None) # fit slownesses coefficients to slowness field    
        return velcoeff.reshape(self.nx,self.ny)

    def convert_pixel_vel_2_basis_slow(self,v): # convert velocity model in pixel basis to equivalent model as slowness coefficients in model basis
        nx,ny = v.shape
        vpert = v-self.vref
        if(np.all(vpert) == 0.): return np.zeros_like(v)
        #coeff = self.coeffs.copy() 
        if(not self.A_calc): 
            A = self.basis_transform_matrix()
            self.A = A
        slowcoeff,res,rank,s = np.linalg.lstsq(self.A.T, 1./vpert.flatten(),rcond=None) # fit slownesses coefficients to slowness field    
        #self.setCoeffs(coeff)
        return slowcoeff.reshape(self.nx,self.ny)

    def convert_pixel_vel_2_basis_vel(self,v): # convert velocity model in pixel basis to equivalent model as velocity coefficients in model basis
        nx,ny = v.shape
        vpert = v-self.vref
        #coeff = self.coeffs.copy() 
        if(not self.A_calc): 
            A = self.basis_transform_matrix()
            self.A = A
        velcoeff,res,rank,s = np.linalg.lstsq(self.A.T, vpert.flatten(),rcond=None) # fit slownesses coefficients to slowness field    
        #self.setCoeffs(coeff)
        return velcoeff.reshape(self.nx,self.ny)
    
    def getCoeffs(self):
        return self.coeffs.copy()
    
    def setCoeffs(self,c):
        assert self.coeffs.shape == c.shape
        self.coeffs = c

    def getSlowness(self,nx=None,ny=None,returncoeff=False):
        ''' With no arguments this will return a slowness field. 
            If bases are 2D pixels the keyword returncoeff is ignored
            If bases are not pixels and returncoeff is False, then a slowness field is evaluated and returned at (nx,ny)
            If bases are not 2D pixels and returncoeff is True, then slowness basis coefficients are returned.
            Default values of (nx,ny) are determined by .getImage() and are the input resolution of the model.
        '''   
        if(self.basis_type == '2Dpixel'):
            if(self.coeff_type=='velocities'):
                if(returncoeff): return 1./(self.vref + self.coeffs) - self.sref # because coefficients are velocities
                return 1./(self.vref + self.coeffs)                              # because coefficients are velocities
            else:
                if(returncoeff): return self.coeffs.copy()      # because coefficients are slownesses
                return self.sref + self.coeffs.copy()            # because coefficients are slownesses
        else:
            if(self.coeff_type=='velocities'):
                if(returncoeff): return self.fitcoefficientsV2S() # we need to find slowness coefficients from velocity coefficients here
                return 1./(self.vref + self.getImage(nx=nx,ny=ny)) # return a slowness field after summation of velocity bases
            else:
                if(returncoeff): return self.coeffs.copy()  # coefficients are slownesses and so we return coefficients
                return self.sref + self.getImage(nx=nx,ny=ny)    # return a slowness field after summation of slowness bases
            
    def setVelocity(self,v):  # set Velocity coefficients
        if(self.coeff_type=='velocities'):
            assert self.coeffs.shape == v.shape
            self.coeffs = v.copy()
        else:
            if(self.basis_type == '2Dpixel'):
                assert self.coeffs.shape == v.shape
                self.coeffs = 1./(v + self.vref) - self.sref
            else:
                print(' Error: can not set velocity coefficients if coeff type are slownesses and basis is not pixel')
                pass # coefficients are slownesses and we need to find and set equivalent velocity coefficients (not implemented) 

    def setSlowness(self,s): # set Slowness coefficients
        if(self.coeff_type=='velocities'):
            if(self.basis_type == '2Dpixel'):
                assert self.coeffs.shape == s.shape
                self.coeffs = 1./(s + self.sref) - self.vref
            else:
                print(' Error: can not set slowness coefficients if coeff type are velocities and basis is not pixel')
                pass # coefficients are velocities and we need to find and set equivalent slowness coefficients (not implemented)
        else:
            assert self.coeffs.shape == s.shape
            self.coeffs = s.copy()    # coefficients are slownesses and so we set new slowness coefficients

    def getImage(self,nx=None,ny=None): # returns 2D image of model at chosen resolution
        if(nx==None): nx = self.nx
        if(ny==None): ny = self.ny
        dx, dy = (self.xmax - self.xmin)/nx, (self.ymax - self.ymin)/ny
        Ym,Xm = np.meshgrid(np.linspace(self.ymin+dy/2,self.ymax-dy/2,ny),np.linspace(self.xmin+dx/2.,self.xmax-dx/2,nx))
        image = np.zeros_like(Xm)
        for j in range(self.basis.nbases): # sum over bases and evaluate model at each pixel in image
            image += (self.coeffs.flatten()[j]*(self.basis.evaluate(j,(Xm,Ym))))
        return image

    def getbasisImage(self,j,nx=None,ny=None): # returns 2D image of model at chosen resolution
        if(nx==None): nx = self.nx
        if(ny==None): ny = self.ny
        dx, dy = (self.xmax - self.xmin)/nx, (self.ymax - self.ymin)/ny
        Ym,Xm = np.meshgrid(np.linspace(self.ymin+dy/2,self.ymax-dy/2,ny),np.linspace(self.xmin+dx/2.,self.xmax-dx/2,nx))
        return self.basis.evaluate(j,(Xm,Ym))

#--------------------------------------------------------------------------------------------
# Other utility functions
#
#--------------------------------------------------------------------------------------------

    def norm(self,x):
        return np.sqrt(x.dot(x))
    
    def normalise(self,x):
        return x/self.norm(x)

    def pngToModel(self,pngfile,nx,ny,bg=1.,sc=1.):
        png = Image.open(pngfile)
        png.load()
        model = sc*(bg+np.asarray(png.convert('L').resize((nx,ny)).transpose(Image.ROTATE_270))/255.)
        return model

    def generateSurfacePoints(self,nPerSide,extent=(0,1,0,1),surface=[True,True,True,True],addCorners=True):
        out = []
        if surface[0]:
            out+=[[extent[0],x] for x in np.linspace(extent[2],extent[3],nPerSide+2)[1:nPerSide+1]]
        if surface[1]:
            out+=[[extent[1],x] for x in np.linspace(extent[2],extent[3],nPerSide+2)[1:nPerSide+1]]
        if surface[2]:
            out+=[[x,extent[2]] for x in np.linspace(extent[0],extent[1],nPerSide+2)[1:nPerSide+1]]
        if surface[3]:
            out+=[[x,extent[3]] for x in np.linspace(extent[0],extent[1],nPerSide+2)[1:nPerSide+1]]
        if addCorners:
            if surface[0] or surface[2]:
                out+=[[extent[0],extent[2]]]
            if surface[0] or surface[3]:
                out+=[[extent[0],extent[3]]]
            if surface[1] or surface[2]:
                out+=[[extent[1],extent[2]]]
            if surface[1] or surface[3]:
                out+=[[extent[1],extent[3]]]
        return np.array(out)

class plot: # This is a set of plotting routines to display 2D velocity models and optionally raypaths and wavefronts on top.
    '''

    A model class containing plot routines for display of 2D velocity/slowness models and optionally raypaths and wavefronts on top.

    '''
    def __init__(self):
        pass
    
    def displayModel(self,model,paths=None,extent=(0,1,0,1),clim=None,cmap=None,
                     figsize=(6,6),title=None,line=1.0,cline='k',alpha=1.0,wfront=None,cwfront='k',
                     diced=True,dicex=8,dicey=8,cbarshrink=0.6,cbar=True,filename=None,**wkwargs):
    
        '''
        
        Function to plot 2D velocity or slowness field
        
        Inputs:
            model, ndarray(nx,ny)           : 2D velocity or slowness field on rectangular grid
            paths, string                   : 
                                    
        '''
        
        plt.figure(figsize=figsize)
        if cmap is None: cmap = plt.cm.RdBu

        # if diced option plot the actual B-spline interpolated velocity used by fmst program
    
        plotmodel = model
        if(diced):
            plotmodel = self.dicedgrid(model,extent=extent,dicex=dicex,dicey=dicey) 
    
        plt.imshow(plotmodel.T,origin='lower',extent=extent,cmap=cmap)


        if paths is not None:
            if(isinstance(paths, np.ndarray) and paths.ndim == 2):
                if(paths.shape[1] == 4): # we have paths from xrt.tracer so adjust
                    paths = self.changepathsformat(paths)

            for p in paths:
                plt.plot(p[:,0],p[:,1],cline,lw=line,alpha=alpha)
    
        if clim is not None: plt.clim(clim)
    
        if title is not None: plt.title(title)
    
        if(wfront is not None):
            nx,ny = wfront.shape
            X, Y = np.meshgrid(np.linspace(extent[0],extent[1],nx), np.linspace(extent[2],extent[3],ny))
            plt.contour(X, Y, wfront.T, **wkwargs)  # Negative contours default to dashed.
    
        if(wfront is None and cbar): plt.colorbar(shrink=cbarshrink)

        if(filename is not None): plt.savefig(filename)
    
        plt.show()

    def dicedgrid(self,v,extent=[0.,1.,0.,1.],dicex=8,dicey=8):    
        nx,ny = v.shape
        x = np.linspace(extent[0], extent[1],nx)
        y = np.linspace(extent[2], extent[3],ny)
        kx,ky=3,3
        if(nx <= 3): kx = nx-1 # reduce order of B-spline if we have too few velocity nodes
        if(ny <= 3): ky = ny-1 
        rect = RectBivariateSpline(x, y, v,kx=kx,ky=ky)
        xx = np.linspace(extent[0], extent[1],dicex*nx)
        yy = np.linspace(extent[2], extent[3],dicey*ny)
        X,Y = np.meshgrid(xx,yy,indexing='ij')
        vinterp = rect.ev(X,Y)
        return vinterp

    def changepathsformat(self,paths):
        p = np.zeros((len(paths),2,2))
        for i in range(len(paths)):
            p[i,0,0] = paths[i,0]
            p[i,0,1] = paths[i,1]
            p[i,1,0] = paths[i,2]
            p[i,1,1] = paths[i,3]
        return p
    
    
