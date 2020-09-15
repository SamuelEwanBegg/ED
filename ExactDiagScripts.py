import numpy as np
import HamBuilder as hb


def ExactDiagonalisation(inputdata):                                                                                                                                                             
    [XXZ,Ising,delta,Jz,Jy,Jx,N,h,PbC] = inputdata                                                                                                                                           
                                                                                                                                                                                             
    Sx = 0.5*np.asarray([[0,1.0],[1.0,0]])                                                                                                                                                   
    Sy = 0.5*np.asarray([[0,-1.0j],[1.0j,0]])                                                                                                                                                
    Sz = 0.5*np.asarray([[1.0,0],[0,-1.0]])                                                                                                                                                  
                                                                                                                                                                                                 
                                                                                                                                                                                                 
    if XXZ == 1:                                                                                                                                                                             
        if N > 2:                                                                                                                                                                        
            H = Jz*hb.tens(Sz,Sz,N,0,PbC) + Jy*hb.tens(Sy,Sy,N,0,PbC) + Jx*hb.tens(Sx,Sx,N,0,PbC) #+ h*hb.tens(Sx,np.identity(2),N,1)                                                
        elif N == 2:                                                                                                                                                                     
            H = Jz*np.kron(Sz,Sz)  + Jy*np.kron(Sy,Sy) + Jx*np.kron(Sx,Sx)# + h*np.kron(np.identity(2),Sx) + h*np.kron(Sx,np.identity(2))                                            
                                                                                                                                                                                                 
    if Ising == 1:                                                                                                                                                                           
        if N > 2:                                                                                                                                                                        
            H = h*hb.tens(Sx,np.identity(2),N,1,0) + Jz*hb.tens(Sz,Sz,N,0,PbC)                                                                                                        
        elif N == 2:                                                                                                                                                                     
            H = Jz*np.kron(Sz,Sz) + h*np.kron(np.identity(2),Sx) + h*np.kron(Sx,np.identity(2))                                                                                       
                                                                                                                                                                                                 
    w,v = np.linalg.eigh(H) 

    return w,v

def wavefun_gen(tvect,evec,evls,ovinit,nummodes,real):   #wavefunction = wavefun(tvec,v,w,st_v_ovI,nummode)  # bec careful initially nummode was total - 1                                                                                                                
        if real == 1:
            rr = -1.0j
        else:
            rr = -1.0 
        wavesum = np.zeros([np.size(tvect),np.size(evec[:,0])],dtype = complex) #time and vector size                                                                                            
        for i in range(0,nummodes):                                                                                                                                                          
                wavesum =   wavesum + np.outer(np.exp(rr*evls[i]*tvect),ovinit[i]*evec[:,i])                                                                                                  
                                                                                                                                                                                                 
        wavefunc = wavesum                                                                                                                                                                       
        return wavefunc                                                                                                                                                                          
def wavefun(tvect,evec,evls,ovinit,nummodes):   #wavefunction = wavefun(tvec,v,w,st_v_ovI,nummode)  # bec careful initially nummode was total - 1                                                                                                                 
        wavesum = np.zeros([np.size(tvect),np.size(evec[:,0])],dtype = complex) #time and vector size                                                                                            
        for i in range(0,nummodes):                                                                                                                                                          
                wavesum =   wavesum + np.outer(np.exp(-1.0j*evls[i]*tvect),ovinit[i]*evec[:,i])                                                                                                  
                                                                                                                                                                                                 
        wavefunc = wavesum                                                                                                                                                                       
        return wavefunc                                                                                                                                                                          
                                                                                                                                                                                                 
def densitymatrix(tvect,evec,wavef):                                                                                                                                                             
        tmax = np.size(tvect)                                                                                                                                                                    
        densitymatr = np.zeros([np.size(tvect),np.size(evec[:,0]),np.size(evec[:,0])],dtype = complex)                                                                                           
                                                                                                                                                                                                 
        for tt in range(0,tmax):                                                                                                                                                                 
                densitymatr[tt,:,:] = np.outer(wavef[tt,:],np.conj(np.transpose(wavef[tt,:])))     #for an outer product the transpose will not make a difference.                                                                                              
                                                                                                                                                                                                 
        return densitymatr                                                                                                                          
def thermal(evec,evls,beta): 
        num = np.size(evec[:,0])                               
        den = np.zeros((num,num),dtype = complex)                                                         
        for i in range(0,num):                                                                                                                     
                den =   den + np.exp(-beta*evls[i])*np.outer(evec[:,i],np.conj(np.transpose(evec[:,i])))                                                                                                  
        return den                                                                  
def red_den(dimA,dimB,densitymat):                                                                                                                                                         
        red_den = np.zeros([dimA,dimA],dtype = complex)                                                                                                                           
        den_tensor = densitymat[:,:].reshape([dimA,dimB,dimA, dimB])                                                            # put in this form as need square dimensions to trace over                                                       
        red_den[:,:] = np.trace(den_tensor, axis1 = 1, axis2 = 3)                                                                                                                                                              
        return red_den                                                                                                                                                                           
                                                                                                                                                                                                 
def Sz():
	zmat = 0.5*np.asarray([[1.0,0],[0,-1.0]])  
	return zmat
                                                                                                                                                                                                
def Sx():
	xmat = 0.5*np.asarray([[0,1.0],[1.0,0]])                                                                                                                                           
	return xmat
	
def Sy():
	ymat = 0.5*np.asarray([[0,-1.0j],[1.0j,0]]) 
	return ymat 
