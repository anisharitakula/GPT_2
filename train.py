import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# read it in to inspect it
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text_length=len(text)
vocab=sorted(list(set(text)))
vocab_size=len(vocab)
split=0.9
block_size=8
batch_size=32
epochs=10
learning_rate=1e-2
max_iters=3000
eval_iters=200

stoi={s:i for i,s in enumerate(vocab)}
itos={i:s for i,s in enumerate(vocab)}

encode=lambda x:[stoi[a] for a in x]
decode=lambda x:''.join([itos[a] for a in x])

data=torch.tensor(encode(text),dtype=torch.long)


train_data=data[:round(split*text_length)]
val_data=data[round(split*text_length):]



torch.manual_seed(1337)
class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding=nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=vocab_size)
    
    def forward(self,idx,targets=None):
        logits=self.token_embedding(idx)
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        for i in range(max_new_tokens):
            logits,loss=self(idx)#(B,T,C)

            logits=logits[:,-1,:]#(B,C)

            probs=F.softmax(logits,dim=1) #(B,C)
            #idx_next=torch.argmax(probs,dim=1).view(-1,1) #(B,1)
            #If I use argmax, I am always getting the same character
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1) #(B,I+1)
        return idx


    

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

model=BigramLanguageModel(vocab_size)
x_batches,y_batches=get_batch('train')
logits,loss=model(x_batches,y_batches)
print(loss)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter%300==0:
        losses=estimate_loss()
        print(f"Step {iter}: The train loss is {losses['train']} and val loss is {losses['val']}")

    x_batches,y_batches=get_batch('train')
    logits,loss=model(x_batches,y_batches)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx=torch.zeros((1,1),dtype=torch.long)
print(decode(model.generate(idx,500)[0].tolist())) 

        
