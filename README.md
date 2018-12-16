# vizdoom_agent

## Dependencies
* ViZDoom: https://github.com/mwydmuch/ViZDoom
* PyTorch: https://pytorch.org/
* OpenAI Gym: https://github.com/openai/gym
* scikit-image: https://scikit-image.org/

## Training
Run `main.py` and specify the result folder (`--result-dir`) where the model is saved along with meta information including parameters, environment configuration and the history of training loss / episodic reward (average in the past 50 finished episodes). For all the arguments / flags for training, go to `arguments.py` for default values and more information.

Some examples for training the model:
* `python3 main.py --result-dir ./perfect_rgb` will train with perfect RGB input.
* `python3 main.py --use-depth --result-dir ./depth_rgb` will train with RGB + depth field input.
* `python3 main.py --disable-rgb --use-depth ./depth_only` wil train with just depth field input.
* `python3 main.py --jitter-rgb --use-depth ./noisy_rgb_depth` wil train with noisy RGB + (normal) depth input. Go to `vizdoom_env.py` if you want to modify the amount (variance) of RGB noise.

## Evaluation
Run `enjoy.py` with the result folder. For example, `python3 enjoy.py ./perfect_rgb` will start runing the model saved in the folder `./perfect_rgb/`. It launches a new window with visualization of the game and prints out the result for every episode in the terminal. You can configure the `--fps` flag to control the speed of simulation, and `--seed` flag for the randomization.

