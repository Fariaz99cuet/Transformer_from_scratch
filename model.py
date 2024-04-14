import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class LayerNormalization(nn.Module):
    def __init__(self, input_feature:int,eps: float=10**-6)->None:
        super().__init__()

        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(input_feature))
        self.bias=nn.Parameter(torch.zeros(input_feature))

    def forward(self,x):
        # x=(batch,seq_len,hidden_size)
        mean=x.mean(dim=-1,keepdim=True) #(batch,batch_seq,1)

        std=x.std(dim=-1,keepdim=True)

        return self.alpha*(x-mean)/(std+self.eps) + self.bias

class FeedForwardNetwork(nn.Module)  :
    def __init__(self,d_model:int, d_ff:int, dropout: float) -> None:
        super().__init__()

        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(d_ff,d_model)

    def forward(self,x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class InputEmbedding(nn.Module):
    def __init__(self,d_model:int,vocab_size:int)->None:
        super().__init__()

        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        #x = (batch,seq_len)--> (batch,seq_len,d_model)
        return self.embedding(x)*math.sqrt(self.d_model)

class PositionEmbedding(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float) ->None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        #PositionEmbedding  vector of shape (seq_len,d_model)

        PosEmb=torch.zeros(self.seq_len,self.d_model)

        #pos of seq_len

        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) #(seq_len,1)

        d=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(1000.0)/d_model)) #(d_model/2)

        PosEmb[:,0::2]=torch.sin(position*d)

        PosEmb[:,1::2]=torch.cos(position*d)

        PosEmb=PosEmb.unsqueeze(0)
        self.register_buffer('PosEmb',PosEmb)


    def forward(self, x):

        x = x + (self.PosEmb[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    def __init__(self,input_feature:int,dropout:float)->None:
        super().__init__()

        self.layernorm=LayerNormalization(input_feature)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,sublayer):
        return x+ self.dropout(sublayer(self.layernorm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float)->None:
        super().__init__()
        self.d_model=d_model
        self.H=h
        self.dropout=nn.Dropout(dropout)

        self.w_q=nn.Linear(d_model,d_model,bias=False)
        self.w_k=nn.Linear(d_model,d_model,bias=False)
        self.w_v=nn.Linear(d_model,d_model,bias=False)
        self.w_O=nn.Linear(d_model,d_model,bias=False)
        self.d_k=d_model//self.H


    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k=query.shape[-1]
        #(B,H,S,d_k)@(B,H,d_k,s)-->
        attention_scores=query@key.transpose(-2,-1)

        if mask is not None:
            attention_scores.masked_fill(mask==0,-1e9)

        attention_scores=attention_scores.softmax(dim=-1)
        #attention_scores=F.softmask(attention_scores,dim=-1)

        if dropout is not None:
            attention_scores=dropout(attention_scores)

        return (attention_scores@value),attention_scores
    def forward(self,q,k,v,mask):
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)

        #(B,S,d_model)->(B,S,H,d_k)
        query=query.view(query.shape[0],query.shape[1],self.H,self.d_k).transpose(1, 2)
        key=key.view(key.shape[0],key.shape[1],self.H,self.d_k).transpose(1, 2)
        value=value.view(value.shape[0],value.shape[1],self.H,self.d_k).transpose(1, 2)

        x,self.attention_scores=MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        #(B,H,S,d_k)->(B,S,H,d_k)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.H * self.d_k)
        #x=x.transpose(1,2).contigous().view(x.shape[0],x.shape[1],self.H*self.d_k)

        return self.w_O(x)




class EncoderBlock(nn.Module):
    def __init__(self,input_feature:int,self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardNetwork,dropout:float)->None:
        super().__init__()
        self.self_attention_block=self_attention_block
        self.residual_connection_block= nn.ModuleList([ResidualConnection(input_feature,dropout) for _ in range(2)])
        self.feed_forward_block=feed_forward_block


    def forward(self,x,src_mask):
        x= self.residual_connection_block[0](x,lambda x:self.self_attention_block(x,x,x,src_mask))
        x= self.residual_connection_block[1](x,self.feed_forward_block)

        return x


class Encoder(nn.Module):
    def __init__(self,input_feature:int,layers=nn.ModuleList)->None:
         super().__init__()
         self.layers=layers
         self.norm=LayerNormalization(input_feature)

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self,input_feature:int,self_attention_block:MultiHeadAttentionBlock,cross_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardNetwork,dropout:float)->None:
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connection_block=nn.ModuleList([ResidualConnection(input_feature,dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x= self.residual_connection_block[0](x,lambda x:self.self_attention_block(x,x,x,src_mask))
        x= self.residual_connection_block[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,tgt_mask))
        x= self.residual_connection_block[2](x,self.feed_forward_block)

        return x

class Decoder(nn.Module):
    def __init__(self,input_features:int,layers:nn.ModuleList)->None:
        super().__init__()
        self.layer=layers
        self.norm=LayerNormalization(input_features)

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layer:
            x=layer(x,encoder_output,src_mask,tgt_mask)

        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self,d_model,vocab_size)->None:
         super().__init__()
         self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embd:InputEmbedding,tgt_embd:InputEmbedding,src_pos:PositionEmbedding,tgt_pos:PositionEmbedding,projection_layer:ProjectionLayer)->None:
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embd=src_embd
        self.tgt_embd=tgt_embd
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.projection_layer=projection_layer

    def encode(self,src,src_mask):
        #(B,S,D)
        src=self.src_embd(src)
        src=self.src_pos(src)

        return self.encoder(src,src_mask)

    def decode(self,encoder_output:torch.Tensor,src_mask=torch.Tensor, tgt=torch.Tensor,tgt_mask=torch.Tensor):
        tgt=self.tgt_embd(tgt)
        tgt=self.tgt_pos(tgt)

        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

    def project(self,x):
        return self.projection_layer(x)


def BuildTransformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,d_model:int=512,h=4,N=4,dropout=0.1,d_ff=2048)->Transformer:

    #create embedding layer
    src_embd=InputEmbedding(d_model,src_vocab_size)
    tgt_embd=InputEmbedding(d_model,tgt_vocab_size)

    # creating positional embedding

    src_pos=PositionEmbedding(d_model,src_seq_len,dropout)
    tgt_pos=PositionEmbedding(d_model,tgt_seq_len,dropout)

    #creating the encoder blocks

    encoder_blocks=[]

    for _ in range(N):
        encoder_attention=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block=FeedForwardNetwork(d_model,d_ff,dropout)
        encoder_block=EncoderBlock(d_model,encoder_attention,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)


    decoder_blocks=[]

    for _ in range(N):
        decoder_self_attention=MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block=FeedForwardNetwork(d_model,d_ff,dropout)
        decoder_block=DecoderBlock(d_model,decoder_self_attention,decoder_cross_attention,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    encoder=Encoder(d_model,nn.ModuleList(encoder_blocks))
    decoder=Decoder(d_model,nn.ModuleList(decoder_blocks))

    projection_layer=ProjectionLayer(d_model,tgt_vocab_size)

    #creating the Final Transformer

    transformer=Transformer(encoder,decoder,src_embd,tgt_embd,src_pos,tgt_pos,projection_layer)

    #Initializing the Parameters
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform(p)

    return transformer


















