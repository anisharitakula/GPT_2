import torch
import torch.nn as nn
from torch.nn import functional as F




split=0.9
block_size=8
batch_size=32
#epochs=10
learning_rate=1e-3
max_iters=5000
eval_interval=500
eval_iters=200
n_embed=32
head_size=16
n_attention_heads=4

torch.manual_seed(1337)

# read it in to inspect it
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab=sorted(list(set(text)))
vocab_size=len(vocab)
text_length=len(text)

stoi={s:i for i,s in enumerate(vocab)}
itos={i:s for i,s in enumerate(vocab)}

encode=lambda x:[stoi[a] for a in x]
decode=lambda x:''.join([itos[a] for a in x])

data=torch.tensor(encode(text),dtype=torch.long)


train_data=data[:int(split*text_length)]
val_data=data[int(split*text_length):]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    loss_dict={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y=get_batch(split)
            logits,loss=model(x,y)
            losses[k]=loss.item()
        loss_dict[split]=losses.mean()
    model.train()
    return loss_dict


class SelfAttention(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embed,head_size,bias=False)
        self.query=nn.Linear(n_embed,head_size,bias=False)
        self.value=nn.Linear(n_embed,head_size,bias=False)


    
    def forward(self,x):
        B,T,C= x.shape
        tril=torch.tril(torch.ones(T,T))
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)

        weights=q@k.transpose(-2,-1)*C**-0.5#(B,T,16)@ (B,16,T)--> (B,T,T)

        weights=weights.masked_fill(tril==0,float('-inf'))
        weights=F.softmax(weights,dim=1)
        out=weights@v#(T,T)@(B,T,HS)--> (B,T,HS)
        return out

class MutiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.multihead=nn.ModuleList([SelfAttention(head_size) for _ in range(num_heads)])
    
    def forward(self,x):
        x=torch.cat([h(x) for h in self.multihead],dim=-1)
        return x

class FeedForward(nn.Module):

    def __init__(self,n_embed):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_embed,n_embed),nn.ReLU(n_embed))
    
    def forward(self, x):
        return self.net(x)

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding=nn.Embedding(vocab_size,
                                          n_embed)
        self.position_embedding=nn.Embedding(block_size,n_embed)
        self.multi_attention=MutiHeadAttention(n_attention_heads,n_embed//n_attention_heads)
        self.ffd=FeedForward(n_embed)
        self.ln1=nn.Linear(n_embed,vocab_size)
    
    def forward(self,idx,targets=None):
        B,T=idx.shape

        
        tok_emb=self.token_embedding(idx)#B,T,n_embed
        pos_emb=self.position_embedding(torch.arange(T))#T,n_embed
        
        
        x=tok_emb + pos_emb #B,T,n_embed
        x=self.multi_attention(x)
        x=self.ffd(x)
        logits=self.ln1(x)
        
        B,T,C=logits.shape
        
        
        if targets is None:
            loss=None
        else:
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        for i in range(max_new_tokens):
            # crop idx to the last block size tokens
            idx_cond=idx[:,-block_size:]

            logits,loss=self(idx_cond)#(B,T,C)
            
            
            logits=logits[:,-1,:]#(B,C)

            probs=F.softmax(logits,dim=-1) #(B,C)
            #idx_next=torch.argmax(probs,dim=1).view(-1,1) #(B,1)
            #If I use argmax, I am always getting the same character
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1) #(B,I+1)
        return idx


    


model=GPTLanguageModel()
# x_batches,y_batches=get_batch('train')
# logits,loss=model(x_batches,y_batches)
# print(loss)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter%eval_interval==0:
        losses=estimate_loss()
        print(f"Step {iter}: The train loss is {losses['train']} and val loss is {losses['val']}")

    x_batches,y_batches=get_batch('train')
    logits,loss=model(x_batches,y_batches)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx=torch.zeros((1,1),dtype=torch.long)
print(decode(model.generate(idx=idx,max_new_tokens=500)[0].tolist())) 

        
