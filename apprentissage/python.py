
import numpy as np
import matplotlib.pyplot as plt
#import time

 
# Fonction logique ET
entres = np.loadtxt ("apprentissage_ET.txt")
entres = entres.astype('int')
w =   np.loadtxt ("POIDS_initiaux_ET.txt") # w = 0.1  0.2  0.05
w_initiaux = w
biais = 1
##1 0 0 0
##1 0 1 0
##1 1 0 0
##1 1 1 1
 

m  = int(len(entres))
t_max = m


#entres = np.c_[np.ones(entres.shape[0]),entres]
biais= biais*np.ones(entres.shape[0])
b=np.transpose(biais)
b=b.astype('int')
d1 = entres.shape #int(len(w))

teta = 0
nu =  0.1
d=d1[1]
 
 
r=  1
x= np.c_[b,entres]
delta = np.zeros((m,d))
taux_rec = 1
w_etape = w
etape = 0
print( "entres   initial = ")
print(  entres)
print("--------------------------------") 
print( "w  initial = ", w)  
print("Biais  =   ",biais)
print("Nu  =   ",nu)
print("--------------------------------")  
#raise

ahmed = 1
while  ahmed == 1 : 
    etape = etape + 1
    t = 0 
    taux_rec = 0
    for t in range(t_max) :
        a =  np.dot(x[t,0:-1], w )
        ax = 0
        if a > teta :
           ax = 1
        if   x[t,d] - ax != 0 :
            for j in range(d) :
                w[j] = w[j] +  nu*(x[t,d]-ax)*x[t,j]    
        else :
            taux_rec = taux_rec + 1
            # Affichage de la droite
        axe_x = entres[0:m,1:d-1]+0.5
        axe_y = entres[0:m,1:d-1]+0.5
      
        for j in range(t+1) :    
                if entres[j,2] == 1 :    
                    plt.plot(entres[j,0],entres[j,1],'x',  c='red')   
                else :
                    plt.plot(entres[j,0],entres[j,1],'x',  c='blue')
        plt.xlabel (" var explicative")
        plt.ylabel (" var expliquÃ©e")
        plt.title (f" Classifieur linÃ©aire.  observation { j }  :  {   entres[j,0:3] } ")
 
        x1= np.linspace(min(axe_x), max(axe_x), 5) 
        if w[2]!= 0 :
          y = -w[0]/w[2] - w[1]/w[2]*x1 
          print("--------------------------------------------------")
          print("Etape:",etape," Biais= ",biais," Nu = ",nu)
          print(" w  =  ",w[0],  w[1],w[2])
 
          plt.plot (x1, y, color ='green', ls ="--")
          plt.grid ()
          plt.show()
           
    if taux_rec == m :
        print("--------------------------------------------------")  
        print("Etape: ",etape,"    w = ", w)
        print(" Taux d'apprentissage= ",100*taux_rec/m,"%")
        break
    
print("--------------------------------------------------")  
 
print("VÃ©rifier si tous les exemples sont bien classÃ©s :  ",np.dot(x[0:m,0:-1], w ))


#  time.sleep(temp)