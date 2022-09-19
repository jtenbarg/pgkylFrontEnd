import numpy as np

def arbitrary(B,V):
	#Basis is b, v x b, and b x (v x b)

	#Handle case for B = (1,0,0)
	if B[1] == 0 and B[2] == 0:
		B[1] = B[0]*1e-24

	rot = np.zeros((3,3))
	magB = np.linalg.norm(B); b = B/magB
	magV = np.linalg.norm(V); v = V/magV

	vxb1 = v[1]*b[2] - v[2]*b[1]
	vxb2 = v[2]*b[0] - v[0]*b[2]
	vxb3 = v[0]*b[1] - v[1]*b[0]

	bxvxb1 = b[1]*vxb3 - b[2]*vxb2
	bxvxb2 = b[2]*vxb1 - b[0]*vxb3
	bxvxb3 = b[0]*vxb2 - b[1]*vxb1

	#Rotation Matrix must be Orthonormal
	mag2=np.sqrt(vxb1**2. + vxb2**2. + vxb3**2)
	mag3=np.sqrt(bxvxb1**2. + bxvxb2**2. + bxvxb3**2)

	#Setup Rotation Matrix
	rot[0] = b

	rot[1,0]=vxb1/mag2
	rot[1,1]=vxb2/mag2
	rot[1,2]=vxb3/mag2

	rot[2,0]=bxvxb1/mag3
	rot[2,1]=bxvxb2/mag3
	rot[2,2]=bxvxb3/mag3

	return rot

def fieldAligned(B):
	#Basis is b, (1,0,0) x b, and b x ((1,0,0) x b)

	#Handle case for B = (1,0,0)
	if B[1] == 0 and B[2] == 0:
		B[1] = B[0]*1.e-24

	rot = np.zeros((3,3)); 
	magB = np.linalg.norm(B); b = B/magB

	#Rotation Matrix must be Orthonormal
	magB2=np.sqrt(b[2]**2.+b[1]**2.)
	magB3=np.sqrt((b[2]**2.+b[1]**2.)**2.+b[0]*b[1]*b[0]*b[1]+b[0]*b[2]*b[0]*b[2])

	#Setup Rotation Matrix
	rot[0] = b

	rot[1,0]=0.
	rot[1,1]=-b[2]/magB2
	rot[1,2]=b[1]/magB2

	rot[2,0]=(b[1]**2.+b[2]**2.)/magB3
	rot[2,1]=-b[0]*b[1]/magB3
	rot[2,2]=-b[0]*b[2]/magB3

	return rot
