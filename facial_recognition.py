import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros,array
from PIL import Image, ImageDraw
from math import *
import glob
from PIL import Image
import os
import shutil
import psutil

kpk = 51 #Vous pouvez modifier cette valeur entre 50 et 70 pour avoir une reconnaissance plus précise

def Apprentissage( Folder) : 
	global kpk
	#1. et 2.
	mat2 = zeros((64,64), float)
	matl = zeros(4096, float)
	matX = zeros((100,4096), float)

	# ********************************* Création de X
	for x in range(0,100):
		my_file = os.path.join(FOLDER, "train/visage ("+str(x+1)+").pgm") 
		deb=my_file
		image_entrée = Image.open(deb)
		mat = np.asarray(image_entrée)
		mat2=mat
		ki = -1
		for i in range(0,64): # va de 0(compris) à 63(compris)
			for j in range(0,64):
				ki=ki+1
				matl[ki] = mat2[i,j]
				
		for y in range(0,4096):
			matX[x,y]=matl[y]

	#3.
	# *********************************************** nu 
	nu = zeros(4096, float)
	for x in range(0,4096) : # initialisation de nu
		nu[x] = 0
	# *Calcul de nu
	for j in range(0,4096):  # addition des 100 matrices lignes
		for i in range(0,100):
			nu[j] = nu[j]+matX[i,j]
	for x in range(0,4096) : # division par 100 pour avoir la moyenne
		nu[x] = nu[x]/100

	# ********************************************** Ecart type
	ect = zeros(4096, float)
	for x in range(0,4096) : # initialisation de ect
		ect[x] = 0
	# Calcul 
	for j in range(0,4096):  # addition des 100 matrices lignes
		for i in range(0,100):
			ect[j] = ect[j] + ( abs(matX[i,j]-nu[j]) * abs(matX[i,j]-nu[j]) )
	for x in range(0,4096) : 
		ect[x] = sqrt(ect[x]/100)

	# ************************************* Calcul matrice Y
	matY = zeros((100,4096), float)
	for i in range(0,100):
		for j in range(0,4096):  
			matY[i,j] = (matX[i,j] - nu[j])/ect[j]	

	# ************************************* nu et ect dans matrice M
	matM = zeros((2,4096), float)
	for j in range(0,4096): 
		matM[0,j]=nu[j]
	for j in range(0,4096): 
		matM[1,j]=ect[j]

	#4.
	# ****************************** décomposition en valeur singulière de Y
	U, s, VT = np.linalg.svd(matY, full_matrices=True)
	V = np.transpose(VT)
	matP=V


	sigma = np.zeros((100,4096 ))
	for i in range(min(100, 4096)):
		sigma[i, i] = s[i]
		a1 = np.dot(U, np.dot(sigma, VT))
	sigmaT = np.transpose(sigma)
	matD = np.dot(sigmaT, sigma)

		#print(np.allclose(matY, a1)) # vérif que la décomposition est bonne 

	#5. k=60

	#6. 
	PK = matP[0:4096,0:kpk]
	Z=np.dot(matY,PK)

	return(matM,PK,Z)
		# *************************** FIN APPRENTISSAGE *****************

def Ajout(myfile1,matM,PK,Z) : 
	global kpk
	xp = zeros(4096, float)
	yp = zeros(4096, float)
	
	tailleZ = str(Z.shape) # (3, 3)
	vir = tailleZ.find(",")
	vir2 =tailleZ.find(")")
	larg=tailleZ[1:vir]
	longu=tailleZ[(vir+2):vir2]
	largeur = int(larg)+1
	longueur = int(longu)
	newZ = zeros((largeur,longueur), float)

	fimg = Image.open(myfile1)
	mat = np.asarray(fimg)
	ki = -1
	for i in range(0,64): # va de 0(compris) à 63(compris)
		for j in range(0,64):
			ki=ki+1
			xp[ki] = mat[i,j]
			

	#2. Normalisation 
	for j in range(0,4096):  
		yp[j] = (xp[j] - matM[0,j])/matM[1,j]
		
	#3. Coord en composantes principales
	zp=np.dot(yp,PK)

# ajout au nouveau Z
	newZ[0:(largeur-1),0:4096] = Z[0:(largeur-1),0:4096]
	newZ[largeur-1,0:4096] = zp

	return(zp,newZ)
	
	# ******************************** Fin de l'ajout ******************

def Reconnaissancefaciale(image,matM,PK,Z) : 
	global kpk
	
	xp = zeros(4096, float)
	yp = zeros(4096, float)

	tailleZ = str(Z.shape)  # (3, 3)
	vir = tailleZ.find(",")
	vir2 =tailleZ.find(")")
	larg=tailleZ[1:vir]
	longu=tailleZ[(vir+2):vir2]
	largeur = int(larg) 		#largeur de Z
	longueur = int(longu)		#longueur de Z

	fimg = Image.open(image)
	mat = np.asarray(fimg)
	ki = -1
	for i in range(0,64): # va de 0(compris) à 63(compris)
		for j in range(0,64):
			ki=ki+1
			xp[ki] = mat[i,j]

	f,ax=plt.subplots(1,2)
	ax[0].imshow(fimg, cmap= 'gray')
	plt.text(-60, -4, "Image source")
	#2. Normalisation 
	for j in range(0,4096):  
		yp[j] = (xp[j] - matM[0,j])/matM[1,j]
		
	#3. Coord en composantes principales
	zp=np.dot(yp,PK)


	#4. Calcul distance 
	# on place un min
	Zmin=0
	iZmin=0
	#on le compare aux autres 
	for i in range(0,kpk) :
			Zmin=Zmin + ( abs(zp[i]-Z[0,i])*abs(zp[i]-Z[0,i]) )
	Zmin=sqrt(Zmin)

	for j in range(1,largeur): 
		Zconcu=0

		for i in range(0,kpk) :
				Zconcu=Zconcu + ( abs(zp[i]-Z[j,i])*abs(zp[i]-Z[j,i]) )
		Zconcu=sqrt(Zconcu)
		if Zconcu<Zmin : 
			Zmin = Zconcu
			iZmin=j

	THIS_FOLDER2 = os.path.dirname(os.path.abspath(__file__))
	my_file2 = os.path.join(THIS_FOLDER2, "train/visage ("+str(iZmin+1)+").pgm")	
	trouve=my_file2
	imageTrouvée = Image.open(trouve)
	ax[1].imshow(imageTrouvée, cmap= 'gray')
	plt.text(-2, -4, "Image la plus ressemblante (indice="+str(iZmin+1)+")")
	plt.show()

def ReconnaissancefacialeFULL(image,matM,PK,Z) : 
	global kpk
	
	xp = zeros(4096, float)
	yp = zeros(4096, float)

	tailleZ = str(Z.shape)  # (3, 3)
	vir = tailleZ.find(",")
	vir2 =tailleZ.find(")")
	larg=tailleZ[1:vir]
	longu=tailleZ[(vir+2):vir2]
	largeur = int(larg) 		#largeur de Z
	longueur = int(longu)		#longueur de Z

	fimg = Image.open(image)
	mat = np.asarray(fimg)
	ki = -1
	for i in range(0,64): # va de 0(compris) à 63(compris)
		for j in range(0,64):
			ki=ki+1
			xp[ki] = mat[i,j]

	f,ax=plt.subplots(1,2)
	ax[0].imshow(fimg, cmap= 'gray')
	plt.text(-60, -4, "Image source")
	#2. Normalisation 
	for j in range(0,4096):  
		yp[j] = (xp[j] - matM[0,j])/matM[1,j]
		
	#3. Coord en composantes principales
	zp=np.dot(yp,PK)


	#4. Calcul distance 
	# on place un min
	Zmin=0
	iZmin=0
	#on le compare aux autres 
	for i in range(0,kpk) :
			Zmin=Zmin + ( abs(zp[i]-Z[0,i])*abs(zp[i]-Z[0,i]) )
	Zmin=sqrt(Zmin)

	for j in range(1,largeur): 
		Zconcu=0

		for i in range(0,kpk) :
				Zconcu=Zconcu + ( abs(zp[i]-Z[j,i])*abs(zp[i]-Z[j,i]) )
		Zconcu=sqrt(Zconcu)
		if Zconcu<Zmin : 
			Zmin = Zconcu
			iZmin=j

	THIS_FOLDER2 = os.path.dirname(os.path.abspath(__file__))
	my_file2 = os.path.join(THIS_FOLDER2, "BDD1/visage ("+str(iZmin+1)+").pgm")	
	trouve=my_file2
	imageTrouvée = Image.open(trouve)
	ax[1].imshow(imageTrouvée, cmap= 'gray')
	plt.text(-2, -4, "Image la plus ressemblante (indice="+str(iZmin+1)+")")
	plt.show()

print("Veuillez patienter pendant le chargement des 100 images du dossier train ( pour faire la matrice de 100 images)...")
yop = 101
THIS_FOLDER1 = os.path.dirname(os.path.abspath(__file__))
my_file1 = os.path.join(THIS_FOLDER1, "train/visage ("+str(yop)+").pgm")	
while os.path.exists(my_file1)==True :
	os.remove(my_file1)
	yop = yop+1
	my_file1 = os.path.join(THIS_FOLDER1, "train/visage ("+str(yop)+").pgm")

FOLDER = os.path.dirname(os.path.abspath(__file__))
mM, mPK, mZ = Apprentissage(FOLDER)
mmZz = zeros((1680,kpk), float)
print("Voulez vous ajouter toutes les images de BDD1 (+1580) à la matrice Z ? ")
fx4 = input("1 = OUI et 2 = NON : ")
if ( str(fx4) == "1" ) : 
	for x in range(0,1680): 			# gère l'ajout de toutes les photos de bdd1 dans la matrice Z
		fx = os.path.join(FOLDER, "BDD1/visage ("+str(x+1)+").pgm") 
		nzp, mmZ = Ajout(fx,mM,mPK,mZ)
		mmZz[x,0:kpk] = mmZ[100,0:kpk]
		
	print("Nous allons maintenant rechercher une image")
	fr2=input("L'image recherchée se trouve dans quelle BDD, 1 ou 2 ? ")
	if ( str(fr2)== "1" ) :
		f = input("Entrez le nom de l'image à reconnaitre (juste le nom de l'image sans le nom du dossier et sans '.pgm') : ")
		THIS_FOLDER1 = os.path.dirname(os.path.abspath(__file__))
		myfile1 = os.path.join(THIS_FOLDER1, "BDD1/"+str(f)+".pgm")

		ReconnaissancefacialeFULL(myfile1,mM,mPK,mmZz)
	if ( str(fr2)== "2" ) :
		f = input("Entrez le nom de l'image à trouver (juste le nom de l'image sans le nom du dossier et sans '.pgm') : ")
		THIS_FOLDER1 = os.path.dirname(os.path.abspath(__file__))
		myfile1 = os.path.join(THIS_FOLDER1, "BDD2/"+str(f)+".pgm")

		ReconnaissancefacialeFULL(myfile1,mM,mPK,mmZz)

if ( str(fx4)!= "1" ) : 
	fr=input("L'image recherchée se trouve dans quelle BDD 1 ou 2 ? ")
	if ( str(fr)== "1" ) :
		f = input("Entrez le nom de l'image à trouver (juste le nom de l'image sans le nom du dossier et sans '.pgm') : ")
		THIS_FOLDER1 = os.path.dirname(os.path.abspath(__file__))
		myfile1 = os.path.join(THIS_FOLDER1, "BDD1/"+str(f)+".pgm")

		Reconnaissancefaciale(myfile1,mM,mPK,mZ)
			
	if ( str(fr)== "2" ) :
		f = input("Entrez le nom de l'image à trouver (juste le nom de l'image sans le nom du dossier et sans '.pgm:') : ")
		THIS_FOLDER1 = os.path.dirname(os.path.abspath(__file__))
		myfile1 = os.path.join(THIS_FOLDER1, "BDD2/"+str(f)+".pgm")

		Reconnaissancefaciale(myfile1,mM,mPK,mZ)


print("Utilisation CPU :",psutil.cpu_percent(),"%")
print("Utilisation RAM :",psutil.virtual_memory().percent,"%")		
		


