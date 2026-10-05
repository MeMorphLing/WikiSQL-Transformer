We built a Transformer from scratch that turns English questions into SQL. 🧮➡️🗄️

No pretrained weights, no nn.Transformer: every attention head, mask and layer written in plain PyTorch,
following "Attention Is All You Need", then trained once on WikiSQL (80k questions, 24k Wikipedia tables).

Results on tables the model has never seen:
• ⟨test execution accuracy⟩% execution accuracy (LSTM seq2seq baseline: ~36%)
• ⟨…⟩% of generated column tokens attend to the right column

What we learned: masks are where bugs hide, warm-up is not optional, and checking the evaluation pipeline with gold
data (99.96% round-trip) before training saves days.

📝 Blog: ⟨Medium link⟩
💻 Code + demo: ⟨GitHub link⟩

#MachineLearning #NLP #Transformers #PyTorch #TextToSQL
