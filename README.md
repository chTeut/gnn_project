# gnn_project

1. Data pre-processing
- Create a document vocab
- Tokenization
- Initial Embedding 
    - token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
- Define Transformer dimension
    - How many Attention heads?
- Train Transformer with document vocab  
    - Loss function 
        - Cross-entropy? 
        - True probability distribution 
         