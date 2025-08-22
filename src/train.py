from __future__ import annotations
import os, sys, argparse, time
import numpy as np
from tqdm import tqdm

# garante que "src" esteja no sys.path quando rodar como módulo
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(ROOT, ".."))
if PROJECT not in sys.path:
    sys.path.append(PROJECT)

from env.flappy_env import FlappyBirdEnv
from ai.dqn_agent import DQNAgent

def train(args):
    env = FlappyBirdEnv(assets_dir=os.path.join(PROJECT, "assets"),
                        render_mode="human" if args.render else None,
                        frame_skip=args.frame_skip)

    state, info = env.reset()
    agent = DQNAgent(
        state_dim=state.shape[0],
        action_dim=env.action_space.n,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        epsilon_decay_steps=args.eps_decay_steps,
        buffer_capacity=args.buffer,
        batch_size=args.batch,
        target_sync=args.target_sync
    )

    best_score = 0
    os.makedirs(os.path.join(PROJECT, "saved_models"), exist_ok=True)
    save_path = os.path.join(PROJECT, "saved_models", "dqn_flappybird.pt")

    ep_bar = tqdm(range(args.episodes), desc="Episodes")
    global_step = 0
    for ep in ep_bar:
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        while not done:
            if args.render:
                env.render()
            action = agent.act(state, explore=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, float(done))
            loss = agent.update()

            state = next_state
            ep_reward += reward
            ep_steps += 1
            global_step += 1

            if args.watch:  # modo assistir não treina
                pass

        score = info.get("score", 0)
        if score > best_score:
            best_score = score
            agent.save(save_path)

        ep_bar.set_postfix(reward=f"{ep_reward:.0f}", score=score, best=best_score, eps=f"{agent.eps:.2f}")

    env.close()
    print(f"Treino finalizado. Melhor score: {best_score}. Modelo salvo em: {save_path}")

def watch(args):
    env = FlappyBirdEnv(assets_dir=os.path.join(PROJECT, "assets"),
                        render_mode="human",
                        frame_skip=args.frame_skip)
    # carrega modelo
    state, info = env.reset()
    agent = DQNAgent(
        state_dim=state.shape[0],
        action_dim=env.action_space.n
    )
    agent.load(args.watch)

    print("Assistindo o agente jogar. Feche a janela para encerrar.")
    try:
        while True:
            state, _ = env.reset()
            done = False
            while not done:
                env.render()
                action = agent.act(state, explore=False)
                state, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
    finally:
        env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay_steps", type=int, default=50_000)
    parser.add_argument("--buffer", type=int, default=100_000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--target_sync", type=int, default=1000)
    parser.add_argument("--watch", type=str, default="")
    args = parser.parse_args()

    if args.watch:
        watch(args)
    else:
        train(args)

if __name__ == "__main__":
    main()
