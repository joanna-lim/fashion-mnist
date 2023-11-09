import torch
import torch.nn as nn

class EmbedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=4, stride=4)  
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 96), requires_grad=True)  
        self.pos_embedding = nn.Parameter(torch.zeros(1, (28 // 4) ** 2 + 1, 96), requires_grad=True)  

    def forward(self, x):
        x = self.conv1(x)  
        x = x.reshape([x.shape[0], 96, -1])  
        x = x.transpose(1, 2) 
        x = torch.cat((torch.repeat_interleave(self.cls_token, x.shape[0], 0), x), dim=1)  
        x = x + self.pos_embedding  
        return x

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_attention_heads = 4
        self.embed_dim = 96
        self.head_embed_dim = self.embed_dim // self.n_attention_heads

        self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)

    def forward(self, x):
        m, s, e = x.shape

        xq = self.queries(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  
        xq = xq.transpose(1, 2)  
        xk = self.keys(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  
        xk = xk.transpose(1, 2) 
        xv = self.values(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim) 
        xv = xv.transpose(1, 2) 

        xq = xq.reshape([-1, s, self.head_embed_dim]) 
        xk = xk.reshape([-1, s, self.head_embed_dim]) 
        xv = xv.reshape([-1, s, self.head_embed_dim])  

        xk = xk.transpose(1, 2)  
        x_attention = xq.bmm(xk)  
        x_attention = torch.softmax(x_attention, dim=-1)

        x = x_attention.bmm(xv)  
        x = x.reshape([-1, self.n_attention_heads, s, self.head_embed_dim])  
        x = x.transpose(1, 2)  
        x = x.reshape(m, s, e)  
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = SelfAttention()
        self.fc1 = nn.Linear(96, 96 * 2)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(96 * 2, 96)
        self.norm1 = nn.LayerNorm(96)
        self.norm2 = nn.LayerNorm(96)

    def forward(self, x):
        x = x + self.attention(self.norm1(x)) 
        x = x + self.fc2(self.activation(self.fc1(self.norm2(x))))  
        return x


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(96, 96)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(96, 10)

    def forward(self, x):
        x = x[:, 0, :]  
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class VisionTransformer16(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbedLayer()
        self.encoder = nn.Sequential(*[Encoder() for _ in range(16)], nn.LayerNorm(96))
        self.norm = nn.LayerNorm(96) 
        self.classifier = Classifier()

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x

class VisionTransformer6(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbedLayer()
        self.encoder = nn.Sequential(*[Encoder() for _ in range(6)], nn.LayerNorm(96))
        self.norm = nn.LayerNorm(96) 
        self.classifier = Classifier()

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x