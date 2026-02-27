# Dashboard

This UI allows users to visulalize the results of runs in aira-dojo.

## Starting the UI

```bash
streamlit run src/dojo/ui/lightweight_dashboard.py
```

### Viewing the dashboard when SSH'd to a cluster

The app prints `http://0.0.0.0:8501`, but that only works on the cluster. To open it in your **local** browser, use SSH port forwarding.

**Option A – Run Streamlit on the login node (simplest)**  
1. Open a **new** SSH session to the cluster and **do not** run `srun`. Stay on the login node.  
2. `cd` to the repo, then run: `streamlit run src/dojo/ui/lightweight_dashboard.py`  
3. On your **laptop** (in a separate terminal), run:
   ```bash
   ssh -L 8501:localhost:8501 as2637@<YOUR_LOGIN_HOST>
   ```
   Replace `<YOUR_LOGIN_HOST>` with the host you `ssh` to (e.g. `unicorn-login-01`).  
4. In your browser on the laptop open: **http://localhost:8501**

**Option B – Streamlit on a compute node (e.g. inside `srun`)**  
1. In the session where Streamlit is running, note the hostname (e.g. `jjs533-compute-01` from the shell prompt).  
2. On your **laptop**, run (replace `LOGIN_HOST` and `COMPUTE_NODE`):
   ```bash
   ssh -L 8501:COMPUTE_NODE:8501 as2637@LOGIN_HOST
   ```
   Example: `ssh -L 8501:jjs533-compute-01:8501 as2637@your-login.cluster.edu`  
3. Open **http://localhost:8501** in your browser.  
If that fails, the cluster may block direct access to compute nodes; use Option A instead.
## Visualizing Runs: The Basics

### Selecting a Meta Experiment
- Select a Base Directory Path (by default it's the logging directory `$LOGGING_DIR/aira-dojo` from your `.env` file) and click "Scan Base Directory".
- Select a Meta Experiment from the dropdown menu.

### Generating and Visualizing a Tree
- "⚙️ Analysis Utilities" > "Select Utility" > "Generate JSON/HTML Trees from Logs" > "Run Utility".
- Go to "🌳 Tree Visualization" Tab and select an experiment from the dropdown menu "Focus on Specific Experiment (Optional)"

### Generating and Inspecting Crash Reports
- "Analysis Utilities" > "Select Utility" > "Generate Crash/Error Reports" > "Run Utility". This will generate a report of all crashes in the selected meta experiment.
- Go to "📁 File Explorer" > "Select File to View" > click on "error_analysis_report.md"

**IMPORTANT**: If you want to genereate crash reports, you must set your `GEMINI_API_KEY` in your `.env` file

### Generating and Inspecting Tree Statistics
- "Analysis Utilities" > "Select Utility" > "Generate Tree Statistics" > "Run Utility". This will generate a report of all tree statistics in the selected meta experiment.
- Go to "📁 File Explorer" > "Select File to View" > select any file starting with "tree_stats/"

**IMPORTANT**: If you want to generate a journal report, you must set your `GEMINI_API_KEY` in your `.env` file


Note: If you don't see any of your files you generated, you may need to refresh the page.
