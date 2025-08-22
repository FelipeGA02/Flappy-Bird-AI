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
Estrutura de pastas

flappybird_ai/
  assets/        # adicione as imagens e sons
  saved_models/  # modelos treinados
  src/
    ai/
    env/
    game/
    train.py
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
0 = não fazer nada

1 = flap / pular

Estrutura de diretórios detalhada
flappybird_ai/
│
├── assets/
│   ├── bird.png
│   ├── pipe.png
│   ├── background.png
│   └── jump.mp3
│
├── saved_models/
│   └── (gerado no treino)
│
└── src/
    ├── __init__.py
    ├── train.py
    ├── game/
    │   ├── __init__.py
    │   ├── flappy_bird.py
    │   └── utils.py
    ├── env/
    │   ├── __init__.py
    │   └── flappy_env.py
    └── ai/
        ├── __init__.py
        ├── dqn_agent.py
        └── model.py