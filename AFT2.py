import torch, math
from torch import nn, einsum
import torch.nn.functional as F

class AFTFull(nn.Module):
    def __init__(self, max_seqlen, dim, hidden_dim=64):
        super().__init__()
        '''
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        dim: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full

        Number of heads is 1 as done in the paper
        '''
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)
        self.wbias = nn.Parameter(torch.Tensor(max_seqlen, max_seqlen))
        nn.init.xavier_uniform_(self.wbias)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x).view(B, T, self.hidden_dim)
        V = self.to_v(x).view(B, T, self.hidden_dim)
        temp_wbias = self.wbias[:T, :T].unsqueeze(0) # sequences can still be variable length

        '''
        From the paper
        '''
        Q_sig = torch.sigmoid(Q)
        temp = torch.exp(temp_wbias) @ torch.mul(torch.exp(K), V)
        weighted = temp / (torch.exp(temp_wbias) @ torch.exp(K))
        Yt = torch.mul(Q_sig, weighted)

        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)

        return Yt

class AFTSimple(nn.Module):
    def __init__(self, max_seqlen, dim, hidden_dim=64):
        super().__init__()
        '''
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        dim: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full
        
        Number of Heads is 1 as done in the paper.
        '''
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x).view(B, T, self.hidden_dim)
        V = self.to_v(x).view(B, T, self.hidden_dim)

        '''
        From the paper
        '''
        weights = torch.mul(torch.softmax(K, 1), V).sum(dim=1, keepdim=True)
        Q_sig = torch.sigmoid(Q)
        Yt = torch.mul(Q_sig, weights)

        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)

        return Yt

class AFTLocal(nn.Module):
    def __init__(self, max_seqlen, dim, hidden_dim=64, s=256):
        super().__init__()
        '''
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        dim: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full
        s: the window size used for AFT-Local in the paper

        Number of heads is 1 as done in the paper
        '''
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)
        self.wbias = nn.Parameter(torch.Tensor(max_seqlen, max_seqlen))
        self.max_seqlen = max_seqlen
        self.s = s
        nn.init.xavier_uniform_(self.wbias)


    def forward(self, x):
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x).view(B, T, self.hidden_dim)
        V = self.to_v(x).view(B, T, self.hidden_dim)
        self.wbias = nn.Parameter(torch.Tensor([
            [self.wbias[i][j] if math.fabs(i-j) < self.s else 0 for j in range(self.max_seqlen)]
            for i in range(self.max_seqlen)
        ]))
        temp_wbias = self.wbias[:T, :T].unsqueeze(0) # sequences can still be variable length

        '''
        From the paper
        '''
        Q_sig = torch.sigmoid(Q)
        temp = torch.exp(temp_wbias) @ torch.mul(torch.exp(K), V)
        weighted = temp / (torch.exp(temp_wbias) @ torch.exp(K))
        Yt = torch.mul(Q_sig, weighted)

        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)

        return Yt

class AFTConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


if __name__ == '__main__':
    layer = AFTFull(
        max_seqlen=20,
        dim=512,
        hidden_dim=200
    )
    # a batch of 64 sequences with 15 timesteps with embed size 512
    x = torch.rand(64, 15, 512)
    y = layer(x)

    print (y.shape) # [64, 15, 512]

