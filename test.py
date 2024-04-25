import torch
from collections import defaultdict

a=torch.tensor([0.1,0.2,0.3,0.4])

print(torch.multinomial(a,num_samples=1))

dict_count=defaultdict(int)
for i in range(1000):
    x=int(torch.multinomial(a,num_samples=1))
    dict_count[x]+=1

print(dict_count)

#Mathematical trick in self.attention
B,T,C=4,8,2
x=torch.randn((B,T,C))
print(x.shape)

xbow=torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        x_prev=x[b,:t+1,:]
        x_mean=x_prev.mean(dim=0)
        xbow[b,t]=x_mean

# print(x[0])
# print(xbow[0])

a=torch.tril(torch.ones(3,3))
a=a/torch.sum(a,dim=1,keepdim=True)
b=torch.randn((3,2))

print(a)
print(b)
print(a@b)
