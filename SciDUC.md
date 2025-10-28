# SciDUC Task Setup

The **SciDUC Benchmark** is designed to evaluate **data efficiency** across multiple domains.  
Each domain (e.g., *WildFin*, *Bucktales*, *Cell Tracking*, *ChimpACT*) includes several sub-tasks representing different annotation ratios (e.g., 1%, 2%, 5%, …, 100%).  
This structure allows testing how well an agent performs under varying levels of supervision.

---

## Directory Structure

```
SciDUC/
├── WildFin/
│ ├── data/ # Raw data (unannotated)
│ ├── k1/ # 1% of annotations
│ │ ├── annotations/ # Annotation files
│ │ ├── src/ # API & dataset setup functions
│ │ ├── evaluate.py # Evaluation function
│ │ └── description.md # Task description and notes
│ ├── k2/ # 2% of annotations
│ ├── k5/ # 5% of annotations
│ ├── k10/ # 10% of annotations
│ ├── k20/ # 20% of annotations
│ └── k100/ # 100% of annotations
├── Bucktales/
├── Cell_Tracking/
├── ChimpACT/
└── Other tasks...
```

---

## Apptainer Environment

Inside the container, the file system visible to the agent is organized as follows:
```
root_dir/
├── work/ # Read-only
│ ├── data/ # All dataset files
│ ├── annotations/ # Task-specific training annotations
│ ├── src/ # Source files accessible as "src.*"
│ ├── evaluate.py # Evaluation script
│ └── description.md # Task description
└── tmp/ # Writable directory for outputs
```

## `evaluate.py`

The **evaluation script** is responsible for running the agent’s program on the dataset and producing a performance score. This file is run in the agent's isolated Apptainer environment. As a result, *all code should be run with respect to the Apptainer file structure, not the SciDUC structure.*

### Expected Behavior

1. **Input**  
   Accepts a `program path` (available as environment variable `PROGRAM_PATH` in Apptainer).

2. **Execution**  
   Evaluates the provided program on the current dataset subset (e.g., `k1`, `k5`, etc.).

3. **Output**  
   Returns a dictionary with the optimization metric under the key `"fitness"`.  
   Example:
   ```python
   {"fitness": 0.823, "details": {...}}
4. **Logging**  
   Print relevant information for debugging and monitoring — the agent can read stdout logs.

### Recommended Header Snippet

To ensure that src imports work correctly, add the following to the top of your evaluate.py file: (This might not be 100% necessary).
```python
import sys, os
# Add the Apptainer bind mount path to Python's module search
sys.path.insert(0, "/work")

print("✅ Added /work to sys.path")
print("Working dir:", os.getcwd())
```

## description.md

This file describes the task and any relevant context or debugging hints.
It should include:

1. The dataset’s purpose and structure

2. The evaluation goal (e.g., mAP, accuracy, IoU, etc.)

3. Notes on known issues (e.g., library incompatibilities)

## Other
I'm thinking about adding a tool_instructions.txt section to help resolve common library related bugs across experiments.