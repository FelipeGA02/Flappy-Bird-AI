# Flappy Bird AI (DQN + Gymnasium + PyTorch)

## Como rodar

1. Crie um ambiente virtual e instale as dependências:
   ```
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   pip install -r requirements.txt

Treino
Rodar treino sem renderização:

python -m src.train --episodes 5000 --render 0
Assistir jogando com modelo salvo:

python -m src.train --watch saved_models/dqn_flappybird.pt --render 1
Recompensas
+1 por passo vivo

+10 por passar um cano

-100 ao colidir

Ações

1 = flap / pular