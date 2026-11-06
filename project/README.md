# DD2603 Course project: RB-Y1

Vision-based imitation learning on the Rainbow Robotics RB-Y1 humanoid robot, in a MuJoCo simulator. The robot
stands in front of a table with RGB-D cameras around it, and the task is to insert a rod into a container. You
design your own policy, which looks at the scene through the cameras and decides how the robot's arm should move.
It is trained from demonstrations and scored in the simulator.

This folder gives you everything else: the simulator, the robot controller, the demonstrations, the vision
encoders, and the scripts to train, run and evaluate a policy. Your own code goes in the `student/` folder,
and that folder is all you hand in.

## Setup

**Your computer must be one of these**, because the robot's software (rby1-sdk) is only made for them:

- a Mac with an Apple chip (M1 or newer) on macOS 14 (Sonoma) or newer;
- Linux with glibc 2.35 or newer, e.g. Ubuntu 22.04 or newer;
- Windows, with Ubuntu 22.04 or newer in WSL. Keep the project in your Linux home folder (`~`), not under `/mnt/c`.

Intel Macs, macOS 13 or older, and older Linux versions (e.g. Ubuntu 20.04) do not work. **If your computer is not
on the list, contact the TAs before you start**, and we will find a solution for you.

You also need **git** and **conda** (e.g. Miniconda). Check that you have both:

```bash
git --version
conda --version
```

If either does not print a version, install it first, following its own instructions.

Then get the course repository and make the project's environment, inside its `project/` folder (every command
in the project, here and in the guide, runs from there):

```bash
git clone https://github.com/GeoRob-Lab/dd2603-course.git
cd dd2603-course/project
conda env create -p ./miniconda_env -f environment.yml
conda activate ./miniconda_env
python -m rbylearn.sim.models          # the robot's model files, about 250 MB, once
```

Last, get the demonstrations (about 400 MB) from the course's GitHub release and unzip them into `datasets/`:

```bash
curl -LO https://github.com/GeoRob-Lab/dd2603-course/releases/download/project-v1.0/insertion_vision.zip
unzip insertion_vision.zip -d datasets/        # -> datasets/insertion_vision/
rm insertion_vision.zip
```

You should see the environment created without errors, and `datasets/insertion_vision/` holding 40 folders,
`demo_0` to `demo_39`. In every new terminal, activate the environment again from the project folder:
`conda activate ./miniconda_env`.

## The guide

**Everything else is in the guide: [open it in your browser](https://raw.githack.com/GeoRob-Lab/dd2603-course/main/project/docs/guide.html).**
You also have it in your clone, as `docs/guide.html`:

```bash
open docs/guide.html        # macOS; on Linux: xdg-open docs/guide.html; on WSL: explorer.exe docs/guide.html
```

It walks you through the simulator, running a first policy, the demonstrations, training, running and
evaluating your own policy, and it lists every input and output your policy can use.
