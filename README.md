# DD2603 Geometric Robot Learning: assignments and project

This repository contains the assignments and the course project of DD2603: the notebooks you work on, and the code and data they need.

| | Topic | Instructions |
|---|---|---|
| Programming Assignment 1 | Manifold geometry for robot poses | [assignments/assignment1](assignments/assignment1/README.md) |
| Programming Assignment 2 | *published later* | |
| Project | *published later* | [project](project/README.md) |

## How the assignments work

* Each assignment is a single Jupyter notebook. You work through it and hand in the executed notebook.
* We recommend running the assignments in **Google Colab**: the link in each assignment's README opens the notebook, and its first cell installs everything it needs. You can also work on your own computer; each assignment's README explains how.
* Each assignment comes with its own small Python package (`assignment1utils`, `assignment2utils`, ...) with helper code, data and checks for your code. The assignments are independent of each other.

## Versions

Each assignment is published as a fixed version, with a git tag such as `assignment1-v1.0`. The links in the assignment READMEs and the notebooks use that version, so what you install does not change while you work. If an assignment needs a correction, it gets a new version (for example `assignment1-v1.1`) and you will be told what to do.

## Repository layout

```
dd2603-course/
├── README.md              this file
├── assignments/
│   ├── assignment1/       notebook, instructions, and the assignment1utils package with its data
│   └── ...                later assignments, one folder each
└── project/               the course project (published later)
```
