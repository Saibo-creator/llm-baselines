from models.llama import RMSNorm, LlamaMLP, LlamaAttention, precompute_freqs_cis
from models.base import GPTBase
from mixture_of_experts import MoE
from torch import nn 
import tiktoken
import torch 
import math
from torch.nn import functional as F

class MixtureOfExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.moe = MoE(
            dim = config.n_embd,            # dimension of model
            num_experts = 8,               # increase the experts (# parameters) of your model without increasing computation
            hidden_dim = config.n_embd * 2,           # size of hidden dimension in each expert, defaults to 4 * dimension
            second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
            second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
            second_threshold_train = 0.2,
            second_threshold_eval = 0.2,
            capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
            capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
            loss_coef = 1e-2                # multiplier on the auxiliary expert balancing auxiliary loss
        )
    
    def forward(self, x):
        return self.moe(x)

class BrainFormerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.attn1 = LlamaAttention(config)
        self.ln2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.mlp1 = LlamaMLP(config)
        self.ln3 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.moe1 = MixtureOfExperts(config)
        self.ln4 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.mlp2 = LlamaMLP(config)
        self.ln5 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.attn2 = LlamaAttention(config)
        self.ln6 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.moe2 = MixtureOfExperts(config)
        self.ln7 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.attn3 = LlamaAttention(config)
        self.ln8 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.mlp3 = LlamaMLP(config) 

    def forward(self, x, freqs_cis):
        x = x + self.attn1(self.ln1(x), freqs_cis)
        x = x + self.mlp1(self.ln2(x))
        moe_out1, aux_loss1 = self.moe1(self.ln3(x))
        x = x + moe_out1
        x = x + self.mlp2(self.ln4(x))
        x = x + self.attn2(self.ln5(x), freqs_cis)
        moe_out2, aux_loss2 = self.moe2(self.ln6(x))
        x = x + moe_out2
        x = x + self.attn3(self.ln7(x), freqs_cis)
        x = x + self.mlp3(self.ln8(x))
        return x, aux_loss1 + aux_loss2

class BrainFormer(GPTBase):
    def __init__(self, config):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # create the token and position embeddings
        self.head_dim = config.n_embd // config.n_head
        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([BrainFormerBlock(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd, eps=config.rmsnorm_eps),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default)
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None, get_logits=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        x = self.transformer.drop(tok_emb)
        freqs_cis = self.freqs_cis.to(x.device)[pos]

        moe_aux_loss = 0
        for block_idx, block in enumerate(self.transformer.h):
            x, aux_loss = block(x, freqs_cis=freqs_cis)
            moe_aux_loss += aux_loss
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            loss += moe_aux_loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        logits = logits if get_logits else None

        return {
            "logits": logits,
            "loss": loss,
        }
    

    def get_parameter_group_specs(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        # need to do import here to avoid circular import (since llama imports from base here)
        from .utils import BLACKLIST_WEIGHT_MODULES

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if fpn not in decay and fpn not in no_decay:
                    if pn.endswith("bias"):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, BLACKLIST_WEIGHT_MODULES):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
                    elif "expert.w" in fpn and isinstance(m, whitelist_weight_modules):
                        decay.add(fpn)
                    else:
                        no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]