import numpy as np
import sys
import pandas as pd
mu = 0
sigma = 1
nBits = 32#subject to change
n = 2**nBits
least = []
phi = []
alpha = []
beta = []
weightMatrix = []
binaryInput=[]
#df=pd.DataFrame()

def challengeGen():
    for i in range(n):
        s = bin(i)[2:].zfill(nBits)
        binaryInput.append(list(s))
        s = s.replace('1','-1')
        s = s.replace('0','1')
        s = s+'1'
        least.append(s)
        
        
def calPhi(): #11111->11111, 111-11->-1-1-111, 11-111->-1-1111
    for i in range(n):
        s = least[i]
        temp=""
        j=0
    
        #for j in range(nBits+1):
        while(j<len(s)):
            if(s[j]=='-'):
                j=j+1
            
            if((s.count('-',j,len(s)))%2==0):
                temp=temp+'1'
            else:
                temp=temp+'-1'
            j=j+1
        phi.append(temp)
        
        
def generateW():
    for i in range(nBits+1):
        p = L[0][i]
        q = L[3][i]
        r = L[2][i]
        s = L[1][i]
        alpha.append((p-q+r-s)/2)
        beta.append((p-q-r+s)/2)
    weightMatrix.append(alpha[0])
    for i in range(1,nBits,1):
        weightMatrix.append(alpha[i]+beta[i-1])
    weightMatrix.append(beta[nBits])
   
response = []
def matMul():
    for i in range(len(phi)):
        s = phi[i]
        phiEach = []
        j=0
        #for j in range(len(s)):
        while(j<len(s)):
            if(s[j] == '-' and s[j+1]=='1'):
                phiEach.append(-1)
                j=j+1
            else:
                phiEach.append(1)
            j=j+1
        #print(phiEach)
        np.transpose(phiEach)
        print(phiEach)
        response.append(np.matmul(weightMatrix,phiEach))


def evalResponse():
    print(response)
    for i in range(len(response)):
        if(response[i]<0):
            response[i] = 1
        else:
            response[i]=0   
        binaryInput[i].append(response[i])
            
def TheRiseOfTheCSV():
    df = pd.DataFrame(binaryInput)
    df.to_csv('Database1.csv')


#def insertToDatabase():


    
L = np.random.normal(mu,sigma,(4,nBits+1))
challengeGen()
calPhi() 
generateW()
matMul()
evalResponse()
TheRiseOfTheCSV()




