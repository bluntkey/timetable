# Exam Timetable Scheduling

A graph-colouring approach to exam timetable scheduling with hall capacity constraints. Courses are represented as vertices, conflicts as edges, and time slots as colours. Each slot can accommodate at most `R` courses, provided no two conflicting courses are scheduled in the same slot.

## Algorithms

| Algorithm | Approach | Optimality |
|---|---|---|
| Welsh–Powell | Sort courses by degree and assign non-conflicting courses to slots, respecting capacity. | Not guaranteed |
| DSatur | Select the course with the highest saturation degree and assign the first valid slot. | Not guaranteed |
| Exact Branch and Bound | Uses backtracking, forward checking, symmetry breaking, and lower bounds to search for an optimal schedule. | Guaranteed if proven before the time limit |

The exact algorithm starts with a greedy solution to establish an upper bound. It uses clique size and the capacity lower bound to reduce the search space. If the time limit is reached before optimality is established, the best solution found is reported without claiming it is optimal.

## Requirements

Install the required Python libraries:

```bash
pip install matplotlib networkx pillow
```

## How to Run

### Option 1: Read Input from a File

```bash
python timetable.py --input sample_input.txt
```

The input file contains `N M R`, followed by `M` conflict edges between courses numbered `1` through `N`.

### Option 2: Generate a Random Graph

```bash
python timetable.py --random 30 10 --density 0.3 --seed 5
```

This generates a graph with 30 courses, a capacity of 10 courses per slot, and a conflict-edge density of 0.3.

### Option 3: Interactive Mode

```bash
python timetable.py
```

Follow the prompts to configure the scheduling problem.

### Useful Options

```bash
python timetable.py --input sample_input.txt --time-limit 30
python timetable.py --input sample_input.txt --no-animation
python timetable.py --input sample_input.txt --no-sweep
```

Run a performance sweep across different graph sizes:

```bash
python timetable.py --random 30 20 --density 0.3 \
  --sweep-sizes 10,20,30,40,50 \
  --sweep-R 20 \
  --sweep-density 0.3 \
  --sweep-trials 3 \
  --outdir results
```

- `--time-limit`: Maximum time for the exact algorithm.
- `--no-animation`: Disable animation generation.
- `--no-sweep`: Disable performance experiments.
- `--outdir`: Specify the output directory.

## Outputs

All generated files are saved in the specified output directory, which defaults to `results`.

| File | Description |
|---|---|
| `timetable.csv` | Timetable organized by slots and halls |
| `timetable.txt` | Text representation of the schedule |
| ![Conflict Graph]`conflict_graph.png` | Graph showing conflicts between courses |
| ![Timetable Grid]`timetable_grid.png` | Visual timetable with consistent colours for each slot |
| ![Algorithm Comparison]`comparison_chart.png` | Comparison of algorithm performance |
| `sweep_results.csv` | Slot counts and execution times across graph sizes |
| ![Assignment Animation]`assignment_animation.gif` | Animation of the final method's assignments |
| ![Assignment Animation Dsatur]`assignment_animation_dsatur.gif` | DSatur assignment animation |

The console also displays the number of slots used by each algorithm, whether optimality was proven, and the resulting timetable.

## Performance Analysis

The performance sweep compares the number of slots and execution time as the number of courses increases.

- **Welsh–Powell:** A simple degree-based heuristic.
- **DSatur:** A saturation-based heuristic that prioritizes highly constrained courses.
- **Exact Branch and Bound:** Searches for an optimal solution, but may require substantially more time on difficult instances.

Exact running time includes clique and lower-bound computation, not just the backtracking phase.

## Key Constraints

- Conflicting courses must be assigned to different time slots.
- Each slot can contain at most `R` courses.
- The exact algorithm reports optimality only when it proves the result.
- If its time limit expires, the best solution found is reported without claiming a proof of optimality.

## Project Goal

This project demonstrates graph colouring, constrained optimization, heuristic scheduling, exact search, and algorithm benchmarking through an exam timetable scheduling problem.

## Authors
1CR24CS164 - Saquib Eqbal
1CR24CS158 - Saksham Abrol
