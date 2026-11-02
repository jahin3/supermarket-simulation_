# Supermarket Checkout Simulation

A time-stepped Python simulation of a supermarket checkout area. It helps decide
how many cashier counters to open and which layout (separate lines, one shared
line, or an express lane) gives the best service at the lowest cost.

## Team 6
- Sumiya Tabassum (0432320005101132)
- Jobair Hossain (0432320005101171)
- Soyeb Hossain Anik (0432320005101150)
- Waliul Haque Nipun (0432320005101147)

## Features
- Poisson customer arrivals with a Gaussian rush-hour peak
- Lognormal basket sizes and item-based service time (5 + 1.4 x items seconds)
- Layouts: separate queues, single shared queue, express lane (10 items or fewer)
- Customer behaviour: balking, reneging (patience) and jockeying
- Metrics: average wait, P95 wait, service level, lost customers, queue length, utilization
- Journey log, Gantt chart, animation, ECDF, boxplot and heatmaps
- 30 repeated runs with 95% confidence intervals, Welch and paired t-tests, Cohen's d
- Cost model (wages + waiting + lost customers) and capacity planner
- Validation against Erlang C with the Allen-Cunneen correction
- Interactive slider dashboard and automatic CSV export

## How to run

### Option 1: Google Colab
1. Upload notebook.ipynb (or paste supermarket_simulation_pro.py into a cell).
2. Click Runtime > Run all.
3. CSV files are downloaded automatically at the end.

### Option 2: Your own computer
1. Install Python 3.9 or newer.
2. Run: pip install -r requirements.txt
3. Run: python supermarket_simulation_pro.py
   (Graphs and the animation display best in Jupyter or Colab.)

## Changing parameters
All settings (arrival rate, rush hour, basket size, costs, patience, number of runs)
are in Section 1 (Settings) at the top of the code. Changing them changes the whole
simulation.

## Output files
- simulation_summary.csv: results of the 5 configurations (seed 42)
- simulation_runs.csv: results of 30 repeated runs
- customer_journey_log.csv: arrival, service start, wait and counter of every customer
- cost_analysis.csv: cost per hour for 1 to 5 counters
- capacity_planner.csv: recommended counters for each arrival rate

## Note
All parameters (costs, patience, basket size, arrival rate) are assumptions, not
real shop measurements.

## Report
The full project report is in the docs folder.
