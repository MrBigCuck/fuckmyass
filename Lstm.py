import numpy as np


def sigmoid(x):

	return 1/(1+np.exp(-x))


def softmax(x):

    exps = np.exp(x)
    return exps / np.sum(exps)
	
	
	
def dsigmoid(x):

	value=sigmoid(x)
	
	return value *(1-value)
	
def dtanh(x):

	return 1.0 - np.tanh(x)**2
	
	




class Lstm:


	def __init__(self,inSize,hiddenSize,learning_rate):
	
		#sizes input,output, concatenation size
		self.hSize=hiddenSize
		self.iSize=inSize
		self.zSize=hiddenSize + inSize
		
		#c_old , h_old
		self.c_old = np.zeros((hiddenSize,1))
		self.h_old = np.zeros((hiddenSize,1))
		
		
		#parameters
		self.Wf=np.random.randn(self.hSize,self.zSize)/np.sqrt(self.zSize/2.0)
		self.Wi=np.random.randn(self.hSize,self.zSize)/np.sqrt(self.zSize/2.0)
		self.Wc=np.random.randn(self.hSize,self.zSize)/np.sqrt(self.zSize/2.0)
		self.Wo=np.random.randn(self.hSize,self.zSize)/np.sqrt(self.zSize/2.0)
		self.Wy=np.random.randn(self.iSize,self.hSize)/np.sqrt(self.iSize/2.0)
		
		#biases
		self.bf=np.zeros((self.hSize,1))
		self.bi=np.zeros((self.hSize,1))
		self.bc=np.zeros((self.hSize,1))
		self.bo=np.zeros((self.hSize,1))
		self.by=np.zeros((self.iSize,1))
		
		# Adagrad gradient update relies on having  memory of sum of squares of the gradients
		self.adaWf=np.zeros((self.hSize,self.zSize))
		self.adaWi=np.zeros((self.hSize,self.zSize))
		self.adaWc=np.zeros((self.hSize,self.zSize))
		self.adaWo=np.zeros((self.hSize,self.zSize))
		self.adaWy=np.zeros((self.iSize,self.hSize))
		
	
		self.adabf=np.zeros((self.hSize,1))
		self.adabi=np.zeros((self.hSize,1))
		self.adabc=np.zeros((self.hSize,1))
		self.adabo=np.zeros((self.hSize,1))
		self.adaby=np.zeros((self.iSize,1))
		
		
		
		#learning rate
		
		self.learning_rate = learning_rate
		
		
	def train(self, x ,y):
	
	

		#initialize=====================================================
		
		
		#intermediate values for all time steps
		
		h = {}#holds state vectors through all time stepss
		c={}#also holds state vectors through all time steps
		x_hat={}#holds encoding representation of x 
		y_hat = {}#holds encoding representation of predicted y (output of network)(unnormalized)
		p={} #holds the normalized probabilties 
		h[-1] = np.copy(self.h_old)
		c[-1] = np.copy(self.c_old)
		
		hf = {}
		hi={}
		ho={}
		hc={}
		
		
		#gradients for backward computation
		dWf =np.zeros_like(self.Wf)
		dWi =np.zeros_like(self.Wi)
		dWc =np.zeros_like(self.Wc)
		dWo =np.zeros_like(self.Wo)
		dWy =np.zeros_like(self.Wy)
		
		dbf = np.zeros_like(self.bf)
		dbi = np.zeros_like(self.bi)
		dbc = np.zeros_like(self.bc)
		dbo = np.zeros_like(self.bo)
		dby = np.zeros_like(self.by)

		
		dc_next = np.zeros_like(self.c_old)
		dh_next = np.zeros_like(self.h_old)
		
		
		
		#forward pass==================================================
		
		loss = 0 #loss accumalator
		
		for t in range(len(x)):
		
			x_input=np.zeros((self.iSize,1))
			x_input[x[t]]=1
			x_hat[t] = np.vstack((h[t-1],x_input))
			
			hf[t]=sigmoid( np.dot(self.Wf,x_hat[t]) + self.bf)
			hi[t]=sigmoid( np.dot(self.Wi,x_hat[t]) + self.bi)
			ho[t]=sigmoid( np.dot(self.Wo,x_hat[t]) + self.bo)
			hc[t]=sigmoid( np.dot(self.Wc,x_hat[t]) + self.bc)
			
			
			
			c[t] =hf[t]*c[t-1] + hi[t]*hc[t]
			h[t] = ho[t]*np.tanh(c[t])
			
			#unnormalized output of the Network
			y_hat[t] = np.dot(self.Wy,h[t])+self.by
			
			#normalized output of the network (which is a probability distrubution )
			p[t] = softmax(y_hat[t])
			
			#adding the losses 
			loss+= - np.log(p[t][y[t],0])
		
		#taking their averages	
		loss/=len(x)	

		#backward pass ==========================================================	
		
		for t in reversed(range(len(x))):
		
			dy = np.copy(p[t])
			
			#gradient of the loss function with respect to the output  
			
			dy[y[t]]-= 1
			
			dWy+= np.dot(dy,h[t].T)
			
			dby+= dy
			
			# added the dh_next here because h is used in the next state and the output
			dh = np.dot(self.Wy.T,dy) + dh_next
			
			# compute gradient for ho n h = ho *tanh(c)
			dho = np.tanh(c[t]) * dh
			dho=dsigmoid(ho[t])*dho
			
			#compute gradient for c in h = ho*tanh(c) 
			dc = ho[t] *dh*dtanh(c[t])
			dc = dc + dc_next
			
			# compute gradient for hf in c = hf*c_old + hi *hc
			
			dhf = c[t-1] *dc
			dhf = dsigmoid(hf[t])*dhf
			
			# compute gradient for hi in c = hf*c_old + hi *hc
			
			dhi = hc[t]*dc
			dhi = dsigmoid(hi[t])*dhi
			
			# compute gradient for hi in c = hf*c_old + hi *hc

			dhc = hi[t]*dc
			dhc = dtanh(hc[t])*dhc
		
		
			#now we compute the gradients for the parameters
			
			dWf+=np.dot(dhf,x_hat[t].T)
			dbf+=dhf
			dxf = np.dot(self.Wf.T,dhf)
			
			dWi+=np.dot(dhi,x_hat[t].T)
			dbi+=dhi
			dxi = np.dot(self.Wi.T,dhi)
			
			
			dWo+=np.dot(dho,x_hat[t].T)
			dbo+=dho
			dxo = np.dot(self.Wo.T,dho)
			
			dWc+=np.dot(dhc,x_hat[t].T)
			dbc+=dhc
			dxc = np.dot(self.Wc.T,dhc)
			
			
			# x was used in different gates so its gradient will be the sum of all gradients in all gates
			
			dx = dxo + dxf +dxi +dxc
			
			#split the concatenation of x to get just the hSize part from it 
			
			dh_next = dx[:self.hSize,:]
			
			#gradient for c_old in c = hf*c_old + hi *hc
			dc_next = hf[t]*dc
		
		#performing parameter update=============================================================================

		for param ,dparam,adaparam in zip([self.Wf,self.Wc,self.Wy,self.Wi,self.Wo,self.bf,self.bc,self.by,self.bi,self.bo],
		[dWf,dWc,dWy,dWi,dWo,dbf,dbc,dby,dbi,dbo],
		[self.adaWf,self.adaWc,self.adaWy,self.adaWi,self.adaWo,self.adabf,self.adabc,self.adaby,self.adabi,self.adabo]):
			
			adaparam+=dparam*dparam
			param+=-self.learning_rate*dparam/np.sqrt(adaparam+1e-8)
			
			
		self.c_old=c[len(x)-1]
		self.h_old=h[len(x)-1]
		
		return loss
		
		
		
	def sample(self,seed,n):

		x = np.zeros((self.iSize,1))
		x[seed]=1
		ixes = []
		
		h_old=self.h_old
		c_old=self.c_old

		for i in range(n):
		
			x_concat = np.vstack((h_old,x))
			
			hf=sigmoid( np.dot(self.Wf,x_concat) + self.bf)
			hi=sigmoid( np.dot(self.Wi,x_concat) + self.bi)
			ho=sigmoid( np.dot(self.Wo,x_concat) + self.bo)
			hc=sigmoid( np.dot(self.Wc,x_concat) + self.bc)
			
			
			
			c_old =hf*c_old + hi*hc
			h_old = ho*np.tanh(c_old)
			
			y = np.dot(self.Wy,h_old) + self.by
			
			p= softmax(y)
			
			ix = np.random.choice(range(self.iSize),p=p.ravel())
			#ix = np.argmax(p)
			x=np.zeros((self.iSize,1))
			x[ix]=1
			
			ixes.append(ix)
			
		return ixes	
	
	def reset_state_vectors(self):
	
		self.c_old = np.zeros((self.hSize,1))
		self.h_old = np.zeros((self.hSize,1))
	
		
		
		
def test():

	#open a text file
	data = open('file.txt','r').read()
	chars = list(set(data))
	data_size ,vocab_size = len(data),len(chars)
	print( 'data has %d characters,%d unique.' %(data_size,vocab_size))
	
	
	#make dictionaries for encoding and decoding characters
	char_to_ix = { ch:i for i,ch in enumerate(chars)}
	ix_to_char = {  i:ch for i,ch in enumerate(chars)}
	
	#initialize the network Lstm 
	lstm = Lstm(vocab_size,100,0.1)
	
	
	seq_length = 25
	
	losses= []
	
	smooth_loss = -np.log(1.0/len(chars))*seq_length
	losses.append(smooth_loss)
	
	n,p=0,0
	while True:
	
		#prepare inputs
		
		if p+seq_length+1 >= len(data) or n==0:
			lstm.reset_state_vectors()
			p=0
			
		
		inputs =[char_to_ix[ch] for ch in data[p:p+seq_length]]
		targets =[char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
		
		
		
		#sample from the model to see the current performance 
		
		if n%100 ==0:
			ixs=lstm.sample(inputs[0],100)
			txt=''.join(ix_to_char[ix] for ix in ixs)
			print('----\n%s\n-----' %(txt,))
			
			
		loss=lstm.train(inputs,targets)
		smooth_loss=smooth_loss*0.999 + loss*0.001
		
		if n%100 ==0:
			print('iter %d,loss:%f' %(n,loss))
			#print(lstm.Wf)
			
			
	
		p+=seq_length
		n+=1
		
	
		
		
		
		