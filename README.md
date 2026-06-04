# CS-421 Machine learning for behavioral data

## Project - GoGymi dataset

### *Authors:* Fatumah binta Doukouré 340969 - Louis Tschanz 315774 - Majandra Garcia 347470

---

This project investigates which observable student behaviors in : *written response quality, AI chatbot interaction patterns, and early platform engagement*, are most predictive of academic success, using data from GoGymi, a Swiss e-learning platform for secondary-school students.

---

**Main research question:** How do the linguistic and semantic characteristics of students' written responses relate to their performance across essay (and text) comprehension assessments?

- **Sub-question 1:** How do the linguistic and semantic characteristics of students' written responses relate to their performance across essay assessment?

- **Sub-question 2:** To what extent does chatbot (GymiTrainer) engagement, in terms of : frequency, interaction intensity, and feedback patterns, influence student performance in quizzes and essays?

- **Sub-question 3:** To what extent can early learning behaviors, such as exercise diversity, session consistency, and content focus, predict which students will struggle or succeed later in their learning progression?
  
- **Fairness question:** Are the predictive models from RQ1-3 equitable across Langzeitgymnasium and Kurzzeitgymnasium students, two tracks that differ systematically in their socioeconomic composition and selection process?

---

## Installation

Clone the repository in the folder of your choice:

```bash
git clone https://github.com/Tournedos/MLBD_2026.git
```

Go into the repository and create and activate a virtual environment:

```bash
python -m venv env_MLBD
```

Then:

```bash
source env_MLBD/bin/activate        # Mac/Linux
env_MLBD\Scripts\activate           # Windows
```

And install the required libraries:

```bash
pip install -r requirements.txt
```

Then create a Jupyter kernel:

```bash
python -m ipykernel install --user --name=env_MLBD --display-name "env_MLBD"
```

---

## How to run

The project is divided into three notebooks, one per sub-question. Run them in order:

### Sub-question 1

jupyter notebook `Q1/1_subquestion.ipynb`

### Sub-question 2

jupyter notebook `Q2/2_subquestion.ipynb`

### Sub-question 3

jupyter notebook `Q3/3_subquestion.ipynb`

### Overall fairness

jupyter notebook `Q4_OverallFairness/overall_fairness.ipynb`

This notebook explores fairness analyses across student subgroups (for example: gender, prior achievement bands, and activity levels). It reproduces the figures and computes fairness metrics used in our write-up, and documents mitigation experiments and subgroup diagnostics. The notebook uses the pre-extracted feature table [Q4_OverallFairness/q3_features.csv](Q4_OverallFairness/q3_features.csv).

--

**Run everything (exact steps)**

1. Create and activate the virtual environment (see Installation).

2. Provide the data folder path. Two options:

- Set an environment variable (preferred for CI/headless runs):

```bash
export GOGYMI_DATA="/absolute/path/to/MLBD_2026/data"
```

- Or copy and edit the configuration template:

```bash
cp config_example.py config.py
# then edit config.py and set DATA_DIR to the full path of the data folder
```

3. Run the notebooks in order (interactive):

```bash
jupyter notebook Q1/1_subquestion.ipynb
jupyter notebook Q2/2_subquestion.ipynb
jupyter notebook Q3/3_subquestion.ipynb
jupyter notebook Q4_OverallFairness/overall_fairness.ipynb
```

4. (Optional) Run notebooks headlessly (execute and write outputs). Example using `nbconvert`:

```bash
jupyter nbconvert --to notebook --execute Q1/1_subquestion.ipynb --output executed_Q1.ipynb
```

--

**Make paths configurable in code / notebooks**

At the top of each notebook, add the following snippet to pick up the data path from the environment or from a local `config.py`:

```python
import os

DATA_DIR = os.environ.get('GOGYMI_DATA')
if DATA_DIR is None:
	try:
		from config import get_data_dir
		DATA_DIR = get_data_dir()
	except Exception:
		raise RuntimeError('Set GOGYMI_DATA or create config.py from config_example.py')

# usage example
students_path = os.path.join(DATA_DIR, 'students.csv')
```

This avoids hardcoded paths and makes the notebooks reproducible across machines.

**Data required**

Place the following files under the directory referenced by `DATA_DIR` (the repository `data/` layout):

- `events/` (0.csv, c.csv, g.csv, m.csv, q.csv, s.csv)
- `comments.csv`, `course_ids.csv`, `essay_feedback.csv`, `essay_results.csv`, `gymitrainer.csv`, `gymitrainer_feedback.csv`, `math_questions.csv`, `math_results.csv`, `pageviews.csv`, `quiz_questions.csv`, `quiz_results.csv`, `students.csv`, `teachers.csv`, `text_questions.csv`, `text_results.csv`

The notebook `Q4_OverallFairness/overall_fairness.ipynb` expects the feature table [Q4_OverallFairness/q3_features.csv](Q4_OverallFairness/q3_features.csv) to be present (it is included in the repository).

---

## Project structure

```bash
MLBD_2026/
├── Q1/
│   ├── part1_mlp_training_loss.png
│   ├── part1_oof_diagnostics.png
│   ├── part1_rf_feature_importances.png
│   ├── part2_top_tfidf_features_ridge.png
│   ├── part3_model_comparison_cv.png
│   ├── 1_subquestion.ipynb  # Notebook for sub-question 1
│   └── helpers.py           # Helper functions for Q1
├── Q2/
│   ├── 1_coverage_performance_tables.png
│   ├── 2_chatbot_feature_correlations.png
│   └── 2_subquestion.ipynb  # Notebook for sub-question 2
├── Q3/
│   ├── Feature_importance_w_cluster.png
│   ├── PCA.jpg
│   ├── 3_subquestion.ipynb  # Notebook for sub-question 3
│   └── utils.py             # Utility functions for Q3
├── Q4_OverallFairness/
│   ├── overall_fairness.ipynb  # Notebook for overall fairness analyses
│   └── q3_features.csv         # Feature table used in fairness analyses
├── data/                    # Raw data (not included in git, see Data section)
│   ├── events/
│   │   ├── 0.csv            # Heartbeat events
│   │   ├── c.csv            # Click events
│   │   ├── g.csv            # GymiTrainer events
│   │   ├── m.csv            # Media events
│   │   ├── q.csv            # Question events
│   │   └── s.csv            # Scroll events
│   ├── comments.csv
│   ├── course_ids.csv
│   ├── essay_feedback.csv
│   ├── essay_results.csv
│   ├── gymitrainer.csv
│   ├── gymitrainer_feedback.csv
│   ├── math_questions.csv
│   ├── math_results.csv
│   ├── pageviews.csv
│   ├── quiz_questions.csv
│   ├── quiz_results.csv
│   ├── students.csv
│   ├── teachers.csv
│   ├── text_questions.csv
│   └── text_results.csv
├── .gitignore
├── config.py
├── GoGymi_Data_Tables_Description.pdf
├── helpers.py
├── README.md
└── requirements.txt
```

---

## Data

The data used in this project comes from the GoGymi platform, a Swiss e-learning environment for secondary-school students in mathematics and German language, to help them for the "maturité" exams.

The dataset is not publicly available and was provided as part of the course CS-421 - Machine Learning for Behavioral Data at EPFL. It is not included in this repository.

To reproduce the results, place the data files in a data/ folder at the root of the repository, following the structure described above.

A full description of the datasets is included in `GoGymi_Data_Tables_Description.pdf`.

---

## Authors & Contributions

- **Sub-question 1 :** Louis Tschanz - Sciper: 315774
- **Sub-question 2 :** Majandra Garcia - Sciper : 347470
- **Sub-question 3 :** Fatumah binta Doukouré - Sciper : 340969
- **Overall-fairness :** Fatumah binta Doukouré, Majandra Garcia and Louis Tschanz
