# CS-421 Machine learning for behavioral data

## Project - GoGymi dataset

### *Authors:* Fatumah binta Doukouré 340969 - Louis Tschanz 315774 - Majandra Garcia 347470

---

This project investigates which observable student behaviors in : *written response quality, AI chatbot interaction patterns, and early platform engagement*, are most predictive of academic success, using data from GoGymi, a Swiss e-learning platform for secondary-school students.

---

**Main research question:** How do the linguistic and semantic characteristics of students' written responses relate to their performance across essay (and text) comprehension assessments?

- **Sub-question 1:** How do the linguistic and semantic characteristics of students' written responses relate to their performance across essay and text comprehension assessments?

- **Sub-question 2:** To what extent does chatbot (GymiTrainer) engagement, in terms of : frequency, interaction intensity, and feedback patterns, influence student performance in quizzes and essays?

- **Sub-question 3:** To what extent can early learning behaviors, such as exercise diversity, session consistency, and content focus, predict which students will struggle or succeed later in their learning progression?

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
├── GoGymi_Data_Tables_Description.pdf
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
