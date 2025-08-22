# IA Flappy Bird (DQN)
Este repositório contém um ambiente minimalista do Flappy Bird e um agente
DQN em PyTorch.
## Como usar
1. Crie o ambiente virtual e instale dependências:
 ```
python -m venv .venv && source .venv/bin/activate # (Linux/macOS)
# ou: .venv\Scripts\activate (Windows)
pip install -r requirements.txt
 ```
2. Treinar:
 ```
python -m src.train --episodes 2000 --render_every 0
 ```
3. Jogar com o modelo treinado (render):
 ```
python -m src.play --checkpoint checkpoints/best.pt --fps 60
 ```
## Notas
- O ambiente segue a API do Gymnasium (obs, reward, terminated, truncated,
info).
- Renderização é opcional durante o treino para acelerar.
- O estado é um vetor com: [dist_x_pipe, dist_y_top, dist_y_bottom, vel_y].
- Ações: 0 = não pular, 1 = pular.
