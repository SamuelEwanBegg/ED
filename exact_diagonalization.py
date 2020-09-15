import matplotlib.pyplot as plt
import scipy.linalg
import scipy as sp
import scipy.sparse.linalg as sp_linalg
import numpy as np
import HamBuilder as hb
import ExactDiagScripts as ED
import copy

stem = '/home/samuel/'

#Inputs
############################################################
tsteps = 50
step = 0.1
tvec = step*np.arange(0,tsteps)
 
N = 8
lengthrow =3
number_rows = 3
a = 0.0

#Initialize Translationally Invariant State (up coefficient, down coefficient)
psi_0 = hb.statebuild_bloch_transinvariant(1.0/np.sqrt(1 + a*np.conj(a)),a/np.sqrt(1 + a*np.conj(a)),N)      
psi_initial = copy.deepcopy(psi_0)

print(np.inner(np.conj(psi_0),psi_0),'checknorm')

#Build Hamiltonian
real = 1 #1 for real time 0 for imaginary time
D1 = 1 #1 is 1D, 0 is custom D and/or long range interactions etc, built from loops (see below)
delta = 1.0
J = -1.0
Jz = delta*J
Jy = -1.3 
Jx = -0.5

hx = -0.0
hy = 0.0
hz = 0.0 
PbC = 0 #periodic boundary conditions

#Variables to Initialise
#######################################
k = 2**N 
nummode = k

Sx = 0.5*np.asarray([[0,1.0],[1.0,0]])                                                                                                                                                                             
Sy = 0.5*np.asarray([[0,-1.0j],[1.0j,0]])                                                                                                                                                                          
Sz = 0.5*np.asarray([[1.0,0],[0,-1.0]])   


st_v_ov = np.zeros([2**N,1],complex)                                                                                                                                                     
Overlapst = np.zeros([tsteps],dtype = complex)                                                                                                  

###############################################################################
#one-dimensional case

if D1 == 1:

        if N == 1:
                H = hx*Sx  + hy*Sy + hz*Sz

        elif N == 2:
                H = Jz*np.kron(Sz,Sz) + hx*np.kron(np.identity(2),Sx) + hx*np.kron(Sx,np.identity(2)) + hy*np.kron(np.identity(2),Sy) + hy*np.kron(Sy,np.identity(2))  + hz*np.kron(np.identity(2),Sz) + hz*np.kron(Sz,np.identity(2))

        else:
                H = Jx*hb.tens(Sx,Sx,N,0,PbC) +  Jy*hb.tens(Sy,Sy,N,0,PbC) + hx*hb.tens(Sx,np.identity(2),N,1,0) + hy*hb.tens(Sy,np.identity(2),N,1,0) +  hz*hb.tens(Sz,np.identity(2),N,1,0)+ Jz*hb.tens(Sz,Sz,N,0,PbC) 

#################################################################################
#two-dimensional case

if D1 == 0:
	Ho = np.zeros([2,2])                                                                                                                                                                                               
	for yy in range(0,N-1):                                                                                                                                                                                        
		Ho = np.kron(Ho,np.zeros([2,2])) # zero matrix for storing final values                                                                                                                                    
																											   
	Hint = Ho                                                                                

	for x in range(0,N):                                                                                                                                                                                           
		y = (x -lengthrow)%N                                                                                                                                                                             
		print(x,y,'Vertical')                                                                                                                                                                                      
		Hint = Hint + Jz*hb.tens_single(Sz,Sz,x,y,N)    
																											   
	for p in range(0,number_rows):                                                                                                                                                                                     
		for i in range(1,lengthrow+1):                                                                                                                                                                             
			x = p*lengthrow + i -1                                                                                                                                                                             
			y = p*lengthrow + 1 + i % lengthrow -1                                                                                                                                                             
			print(x,y, 'Horizontal')      
			Hint = Hint + Jz*hb.tens_single(Sz,Sz,x,y,N)    

	H = Hint + hx*hb.tens(Sx,np.identity(2),N,1,0)
                                                                                                                          
###############################################################################
#perform the diagonlization

w,v = np.linalg.eigh(H) 

################################################################################
#Collect overlaps, normalize eigenvectors, for calculating observables

if real == 1:
	r = 1.0j
else:
	r = 1.0
for i in range(0,k):
	v[:,i] = v[:,i]/sum(v[:,i]*np.conj(v[:,i]))
	st_v_ov[i] = np.inner(np.conj(v[:,i]),psi_0)
	
st_v_ovNORM = np.conj(st_v_ov)*st_v_ov

#Check that squared overlaps add up to 1
print(sum(st_v_ovNORM),'Should be 1, overlap when t = 0')

sz = ED.Sz()
sy = ED.Sy()
sx = ED.Sx()


if N > 1:

    # Initial Observables Operators
    magMatrixZ = hb.tens(sz,np.identity(2),N,1,0)
    magMatrixX = hb.tens(sx,np.identity(2),N,1,0)
    magMatrixY = hb.tens(sy,np.identity(2),N,1,0)
    corrMat = hb.tens(sz,sz,N,0,PbC)
    corrMatX = hb.tens(sx,sx,N,0,PbC)

    NA = int(N/2)
    dimsA = 2**NA
    NB = N - NA
    dimsB = 2**NB


    # Initial Observables Measurement `Outputs'
    energy = np.zeros([tsteps,1],dtype = complex)
    magnetisationZ = np.zeros([tsteps,1],dtype = complex)
    magnetisationX = np.zeros([tsteps,1],dtype = complex)
    magnetisationY = np.zeros([tsteps,1],dtype = complex)
    normalisation = np.zeros([tsteps,1],dtype = complex)
    entropy = np.zeros([tsteps,1],dtype = complex)
    Overlap = np.zeros([tsteps,1],dtype = complex)
    correlation = np.zeros([tsteps,1],dtype = complex)
    correlationX = np.zeros([tsteps,1],dtype = complex)
    magtimeav = np.zeros([tsteps,1])

wavefunction = []

#calculate observables at every time
for ii in range(0,np.size(tvec)):
    print(ii)
   
    if N == 1:

        wavefunction = wavefunction + [ED.wavefun_gen(tvec[ii],v,w,st_v_ov,nummode,real)]
    else:
        wavefunction = ED.wavefun_gen(tvec[ii],v,w,st_v_ov,nummode,real)
        densityMatrix = ED.densitymatrix(tvec[ii],v,wavefunction)
        reduced_density = ED.red_den(dimsA,dimsB,densityMatrix[0,:,:]) 


    if N > 1:
    #Overlap = Overlap + [np.inner(np.conj(np.transpose(psi_0)),wavefunction)] 
        Overlap[ii] = np.inner(np.conj(np.transpose(psi_initial)),wavefunction)
        normalisation[ii] = np.trace(densityMatrix[0,:,:])
        magnetisationZ[ii] = (1.0/float(N))*np.trace(np.dot(magMatrixZ,densityMatrix[0,:,:])/np.trace(densityMatrix[0,:,:]), axis1 = 0, axis2 = 1) #first axis is time 
        magnetisationY[ii] = (1.0/float(N))*np.trace(np.dot(magMatrixY,densityMatrix[0,:,:])/np.trace(densityMatrix[0,:,:]), axis1 = 0, axis2 = 1) #first axis is time 
        magnetisationX[ii] = (1.0/float(N))*np.trace(np.dot(magMatrixX,densityMatrix[0,:,:])/np.trace(densityMatrix[0,:,:]), axis1 = 0, axis2 = 1) #first axis is time 
        correlation[ii] = (1.0/float(N-1+PbC))*np.trace(np.dot(corrMat,densityMatrix[0,:,:])/np.trace(densityMatrix[0,:,:]), axis1 = 0, axis2 = 1) #first axis is time 
        correlationX[ii] = (1.0/float(N-1+PbC))*np.trace(np.dot(corrMatX,densityMatrix[0,:,:])/np.trace(densityMatrix[0,:,:]), axis1 = 0, axis2 = 1) #first axis is time 
        entropy[ii] = -np.trace(np.dot(reduced_density,scipy.linalg.logm(reduced_density)))  


for ii in range(1,np.size(tvec)):
    if N > 1:
        energy[ii] = -(1.0/(N*(ii)*step))*np.log(Overlap[ii]) 

#save outputs to desktop and plot
if N > 1:
    np.save(stem + 'Desktop/recentX_plat',magnetisationX[np.size(magnetisationZ)-1])
    np.save(stem + 'Desktop/recentY_plat',magnetisationY[np.size(magnetisationZ)-1])
    np.save(stem + 'Desktop/recentZ_plat',magnetisationZ[np.size(magnetisationZ)-1])

    np.save(stem + 'Desktop/recentX',magnetisationX)
    np.save(stem + 'Desktop/overlap',Overlap)
    np.save(stem + 'Desktop/normalisation',normalisation)
    np.save(stem + 'Desktop/recentY',magnetisationY)
    np.save(stem + 'Desktop/recentZ',magnetisationZ)
    np.save(stem + 'Desktop/recentcorr',correlation)
    np.save(stem + 'Desktop/recentcorrX',correlationX)
    np.save(stem + 'Desktop/entropy',entropy)
    np.save(stem + 'Desktop/energy',energy)

    plt.plot(tvec,magnetisationX,label = 'X')
    plt.plot(tvec,magnetisationY,label = 'Y')
    plt.plot(tvec,magnetisationZ,label = 'Z')
    plt.plot(tvec,correlation,label = 'zz+1')
    plt.plot(tvec[1::],energy[1::],label = 'energy')
    plt.plot(tvec,Overlap*np.conj(Overlap),'x',label = 'Probability Overlap')
    plt.plot(tvec,normalisation,label = 'Normalisation')
    plt.legend()
    plt.ylim(-2,2)
    plt.show()

